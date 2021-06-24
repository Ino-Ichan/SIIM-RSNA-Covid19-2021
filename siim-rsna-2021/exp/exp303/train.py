import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import cv2

import albumentations
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from tqdm import tqdm
import argparse
import os, sys, yaml

sys.path.append('/workspace/siim-rsna-2021')
from src.logger import setup_logger, LOGGER
from src.meter import mAPMeter, AUCMeter, APMeter, AverageValueMeter
from src.utils import plot_sample_images

# import neptune.new as neptune
import wandb
import pydicom

import time
from contextlib import contextmanager

import timm

import warnings

target_columns = [
    "Negative for Pneumonia", "Typical Appearance", "Indeterminate Appearance", "Atypical Appearance", "is_none"
]


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_torch(seed=516):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def ousm_loss(error, k=2):
    # ousm, drop large k sample
    bs = error.shape[0]
    if len(error.shape) == 2:
        error = error.mean(1)
    _, idxs = error.topk(bs - k, largest=False)
    error = error.index_select(0, idxs)
    return error


# Freeze batchnorm 2d
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


# =============================================================================
# Model
# =============================================================================

class Net(nn.Module):
    def __init__(self, name="resnest101e"):
        super(Net, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=len(target_columns))

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x

# =============================================================================
# Dataset
# =============================================================================

class CustomDataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 transform=None,
                 mode="train",
                 clahe=False,
                 mix=False,
                 use_npy=False,
                 ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform

        self.mode = mode
        self.clahe = clahe
        self.mix = mix
        if self.clahe or self.mix:
            self.clahe_transform = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(16, 16))

        self.cols = target_columns
        self.use_npy = use_npy

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if self.use_npy:
            # images = np.load(row.npy_path)
            images = cv2.imread(row.npy_path)
        else:
            images = pydicom.read_file(row.dicom_path).pixel_array

        if self.clahe:
            single_channel = images[:, :, 0].astype(np.uint8)
            single_channel = self.clahe_transform.apply(single_channel)
            images = np.array([
                single_channel,
                single_channel,
                single_channel
            ]).transpose(1, 2, 0)
        elif self.mix:
            single_channel = images[:, :, 0].astype(np.uint8)
            clahe_channel = self.clahe_transform.apply(single_channel)
            hist_channel = cv2.equalizeHist(single_channel)
            images = np.array([
                single_channel,
                clahe_channel,
                hist_channel
            ]).transpose(1, 2, 0)

        if self.transform is not None:
            images = self.transform(image=images)['image'] / 255                
        else:
            images = images.transpose(2, 0, 1)

        label = row[self.cols].values.astype(np.float16)
        return {
            "image": torch.tensor(images, dtype=torch.float),
            # "image": images,
            "target": torch.tensor(label, dtype=torch.float)
        }


# =============================================================================
# one epoch
# =============================================================================

def train_one_epoch(train_dataloader, model, device, criterion, use_amp, wandb, meters_dict, mode="train"):

    train_time = time.time()
    LOGGER.info("")
    LOGGER.info("+" * 30)
    LOGGER.info(f"+++++  Epoch {e} at CV {cv}")
    LOGGER.info("+" * 30)
    LOGGER.info("")
    progress_bar = tqdm(train_dataloader)

    model.train()
    torch.set_grad_enabled(True)

    # freeze bach norm
    if freeze_bn:
        model = model.apply(set_bn_eval)

    # reset metrics
    for m in meters_dict.values():
        m.reset()

    for step_train, data in enumerate(progress_bar):
        if debug:
            if step_train == 2:
                break

        inputs = data["image"].to(device)
        target = data["target"].to(device)

        bs = inputs.shape[0]

        with autocast(enabled=use_amp):
            output = model(inputs)
            loss = criterion(output, target).mean()

        if accumulation_steps > 1:
            loss_bw = loss / accumulation_steps
            scaler.scale(loss_bw).backward()
            if (step_train + 1) % accumulation_steps == 0 or step_train == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        meters_dict["loss"].add(loss.item(), n=bs)
        meters_dict["AP"].add(output=output.detach(), target=target)
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {meters_dict['loss'].value()[0]}")

    LOGGER.info(f"Train loss: {meters_dict['loss'].value()[0]}")
    LOGGER.info(f"Train mAP: {meters_dict['AP'].value().mean()}")
    LOGGER.info(f"Train time: {(time.time() - train_time) / 60:.3f} min")

    wandb.log({
        f"epoch": e,
        f"Loss/train_cv{cv}": meters_dict['loss'].value()[0],
        f"mAP/train_cv{cv}": meters_dict['AP'].value().mean(),
        f"mAP_metrics/train_cv{cv}": (2 * meters_dict['AP'].value().mean()) / 3,
    })


def val_one_epoch(val_dataloader, model, device, wandb, meters_dict, mode="val"):

    val_time = time.time()
    progress_bar = tqdm(val_dataloader)

    model.eval()
    torch.set_grad_enabled(False)

    # reset metrics
    for m in meters_dict.values():
        m.reset()

    for step_val, data in enumerate(progress_bar):
        if debug:
            if step_val == 2:
                break

        inputs = data["image"].to(device)
        target = data["target"].to(device)

        bs = inputs.shape[0]

        with torch.no_grad():
            output = model(inputs)
            loss = criterion(output, target).mean()

        meters_dict["loss"].add(loss.item(), n=bs)
        meters_dict["AP"].add(output=output, target=target)
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {meters_dict['loss'].value()[0]}")

    LOGGER.info(f"Val loss: {meters_dict['loss'].value()[0]}")
    LOGGER.info(f"Val mAP: {meters_dict['AP'].value().mean()}")
    LOGGER.info(f"Val mAP score: {(2 * meters_dict['AP'].value().mean()) / 3}")
    LOGGER.info(f"Val time: {(time.time() - val_time) / 60:.3f} min")

    log_dict = {
        f"epoch": e,
        f"Loss/val_cv{cv}": meters_dict['loss'].value()[0],
        f"mAP/val_cv{cv}": meters_dict['AP'].value().mean(),
        f"mAP_metrics_old/val_cv{cv}": (2 * meters_dict['AP'].value()[:4].mean()) / 3
    }
 
    for n_t, t in enumerate(target_columns):
        log_dict[f"AP_{t}/val_cv{cv}"] = meters_dict['AP'].value()[n_t]
    wandb.log(log_dict)

    return meters_dict['AP'].value().mean()


# def get_train_transforms(image_size):
#     return albumentations.Compose([
#         albumentations.Transpose(p=0.5),
#         albumentations.VerticalFlip(p=0.5),
#         albumentations.HorizontalFlip(p=0.5),
#         albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
#                                                 brightness_by_max=False, p=0.5),
#         albumentations.Blur(blur_limit=7, p=0.5),
#         # albumentations.HueSaturationValue(p=0.5),
#         albumentations.CenterCrop(540, 540, p=1),
#         albumentations.Resize(image_size, image_size),
#         # albumentations.RandomResizedCrop(height=image_size, width=image_size, scale=(0.08, 1)),
#         albumentations.CoarseDropout(max_holes=3, max_height=50, max_width=50),
#         ToTensorV2()
#     ])
#
#
# def get_val_transforms(image_size):
#     return albumentations.Compose([
#         albumentations.CenterCrop(540, 540, p=1),
#         albumentations.Resize(image_size, image_size),
#         # albumentations.RandomResizedCrop(height=image_size, width=image_size, scale=(0.08, 1)),
#         ToTensorV2()
#     ], p=1.0)


def get_train_transforms(image_size):
    return albumentations.Compose([
           albumentations.ShiftScaleRotate(p=0.5),
           albumentations.Resize(image_size, image_size),
           albumentations.HorizontalFlip(p=0.5),
           albumentations.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
           albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.5),
           albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
        #    albumentations.OneOf([
        #        albumentations.OpticalDistortion(distort_limit=1.0),
        #        albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        #        albumentations.ElasticTransform(alpha=3),
        #    ], p=0.2),
           albumentations.OneOf([
               albumentations.GaussNoise(var_limit=[10, 50]),
               albumentations.GaussianBlur(),
               albumentations.MotionBlur(),
            #    albumentations.MedianBlur(),
           ], p=0.1),
        #   albumentations.OneOf([
        #       albumentations.augmentations.transforms.JpegCompression(),
        #       albumentations.augmentations.transforms.Downscale(scale_min=0.1, scale_max=0.15),
        #   ], p=0.2),
        #   albumentations.imgaug.transforms.IAAPiecewiseAffine(p=0.2),
        #   albumentations.imgaug.transforms.IAASharpen(p=0.2),
          albumentations.RandomResizedCrop(512, 512, scale=(1, 1), p=1),
          albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=5, p=0.5),
        #   albumentations.Normalize(
        #       mean=[0.485, 0.456, 0.406],
        #       std=[0.229, 0.224, 0.225],
        #   ),
          ToTensorV2(p=1)
])


def get_train_transforms2(image_size):
    return albumentations.Compose([
           albumentations.ShiftScaleRotate(rotate_limit=30, p=0.5),
           albumentations.RandomResizedCrop(image_size, image_size, scale=(0.7, 1), p=1),
           albumentations.HorizontalFlip(p=0.5),
           albumentations.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
           albumentations.OneOf([
               albumentations.GaussNoise(),
               albumentations.MotionBlur(blur_limit=3),
           ], p=0.1),
          albumentations.Resize(image_size, image_size),
          albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=2, p=0.5),
          ToTensorV2(p=1)
])


def get_val_transforms(image_size):
    return albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        # albumentations.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
        ToTensorV2(p=1)
])


if __name__ == "__main__":
    print('Start!!!')
    warnings.simplefilter('ignore')

    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('-y', '--yaml_path', type=str,
                        help='configを書いたyamlのPath。例）-y ../config/exp0001.yaml')

    args = parser.parse_args()

    yaml_path = args.yaml_path
    yaml_path = args.yaml_path
    if os.path.isfile(yaml_path):
        with open(yaml_path) as file:
            cfg = yaml.safe_load(file.read())
    else:
        print('Error: No such yaml file')
        sys.exit()
    # seed_everythin
    seed_torch()

    # output
    exp_name = cfg["exp_name"]  # os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join("/workspace/output", exp_name)
    # path
    model_path = output_path + "/model"
    plot_path = output_path + "/plot"
    oof_path = output_path + "/oof"
    sample_img_path = output_path + "/sample_img"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path + "/log", exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(oof_path, exist_ok=True)
    os.makedirs(sample_img_path, exist_ok=True)

    # logger
    log_path = os.path.join(output_path, "log/log.txt")
    setup_logger(out_file=log_path)
    LOGGER.info("config")
    LOGGER.info(cfg)
    LOGGER.info('')

    debug = cfg["debug"]
    if debug:
        LOGGER.info("Debug!!!!!")

    # params
    device_id = cfg["device_id"]
    try:
        device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
    except Exception as e:
        LOGGER.info('GPU is not available, {}'.format(e))
        sys.exit()

    print(device)

    #######################################
    ## params
    #######################################
    model_name = cfg["model_name"]
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]
    n_workers = cfg["n_workers"]
    n_epochs = cfg["n_epochs"]
    start_epoch = cfg["start_epoch"]
    transform = cfg["transform"]
    hold_out = cfg["hold_out"]
    accumulation_steps = cfg["accumulation_steps"]
    early_stopping_steps = cfg["early_stopping_steps"]
    freeze_bn = cfg["freeze_bn"]

    use_amp = cfg["use_amp"]
    use_npy = cfg["use_npy"]

    clahe = cfg["clahe"]
    mix = cfg["mix"]

    #######################################
    ## CV
    #######################################
    df = pd.read_csv(cfg["df_train_path"])

    cv_list = hold_out if hold_out else [0, 1, 2, 3, 4]
    oof = np.zeros((len(df), len(target_columns)))
    best_eval_score_list = []

    for cv in cv_list:

        LOGGER.info('# ===============================================================================')
        LOGGER.info(f'# Start CV: {cv}')
        LOGGER.info('# ===============================================================================')

        # wandb
        wandb.init(config=cfg, tags=[cfg['exp_name'], f"cv{cv}", model_name],
                   project='siim-rsna-covid19-2021-2', entity='inoichan',
                   name=f"{cfg['exp_name']}_cv{cv}_{model_name}", reinit=True)

        df_train = df[df.cv != cv].reset_index(drop=True)
        df_val = df[df.cv == cv].reset_index(drop=True)
        val_index = df[df.cv == cv].index

        #######################################
        ## Dataset
        #######################################
        # transform
        train_transform = get_train_transforms(img_size)
        val_transform = get_val_transforms(img_size)

        train_dataset = CustomDataset(df=df_train, image_size=img_size, clahe=clahe, mix=mix,
                                      transform=train_transform, use_npy=use_npy, mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      pin_memory=False, num_workers=n_workers, drop_last=True)
        # plot sample image
        # plot_sample_images(train_dataset, sample_img_path, "train", normalize="imagenet")
        plot_sample_images(train_dataset, sample_img_path, "train", normalize=None)

        val_dataset = CustomDataset(df=df_val, image_size=img_size, clahe=clahe, mix=mix,
                                    transform=val_transform, use_npy=use_npy, mode="val")
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    pin_memory=False, num_workers=n_workers, drop_last=False)

        # plot_sample_images(val_dataset, sample_img_path, "val",  normalize="imagenet")
        plot_sample_images(val_dataset, sample_img_path, "val", normalize=None)

        # ==== INIT MODEL
        device = torch.device(device)
        model = Net(model_name).to(device)

        optimizer = optim.Adam(model.parameters(), lr=float(cfg["initial_lr"]), eps=1e-7)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=float(cfg["final_lr"]))

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        scaler = GradScaler(enabled=use_amp)

        # load weight
        load_checkpoint = cfg["load_checkpoint"][cv]
        LOGGER.info("-" * 10)
        if os.path.exists(load_checkpoint):
            weight = torch.load(load_checkpoint, map_location=device)
            model.load_state_dict(weight["state_dict"])
            LOGGER.info(f"Successfully loaded model, model path: {load_checkpoint}")
            optimizer.load_state_dict(["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        else:
            LOGGER.info(f"Training from scratch..")
        LOGGER.info("-" * 10)

        # wandb misc
        wandb.watch(model)

        # ==== TRAIN LOOP

        best = -1
        best_epoch = 0
        early_stopping_cnt = 0

        meters_dict = {
            "loss": AverageValueMeter(),
            "AP": APMeter(),
        }

        for e in range(start_epoch , start_epoch + n_epochs):
            if e > 0:
                wandb.log({
                    "Learning Rate": optimizer.param_groups[0]["lr"],
                    "epoch": e
                })
                train_one_epoch(train_dataloader, model, device, criterion, use_amp, wandb, meters_dict)

            score = val_one_epoch(val_dataloader, model, device, wandb, meters_dict)
            scheduler.step()

            LOGGER.info('Saving last model ...')
            model_save_path = os.path.join(model_path, f"cv{cv}_weight_checkpoint_last.pth")

            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, model_save_path)

            if best < score:
                LOGGER.info(f'Best score update: {best:.5f} --> {score:.5f}')
                best = score
                best_epoch = e

                LOGGER.info('Saving best model ...')
                model_save_path = os.path.join(model_path, f"cv{cv}_weight_checkpoint_best.pth")

                torch.save({
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, model_save_path)

                early_stopping_cnt = 0
            else:
                # early stopping
                early_stopping_cnt += 1
                if early_stopping_cnt >= early_stopping_steps:
                    LOGGER.info(f"Early stopping at Epoch {e}")
                    break

            LOGGER.info('-' * 20)
            LOGGER.info(f'Best val score: {best}, at epoch {best_epoch} cv{cv}')
            LOGGER.info('-' * 20)

            best_eval_score_list.append(best)
            wandb.log({
                "Best mAP": best,
                # "Best mAP metrics": (2 * best) / 3,
            })

    #######################################
    ## Save oof
    #######################################
    mean_score = np.mean(best_eval_score_list)
    LOGGER.info('-' * 20)
    LOGGER.info(f'Oof score: {mean_score}')
    LOGGER.info('-' * 20)

