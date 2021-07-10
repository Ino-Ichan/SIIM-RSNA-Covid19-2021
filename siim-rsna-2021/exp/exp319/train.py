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
        enet = timm.create_model(name, True, drop_path_rate=0.2)
        self.b0 = nn.Sequential(
            enet.conv_stem,
            enet.bn1,
            enet.act1,
        )
        self.b1 = enet.blocks[0]
        self.b2 = enet.blocks[1]
        self.b3 = enet.blocks[2]
        self.b4 = enet.blocks[3]
        self.b5 = enet.blocks[4]
        self.b6 = enet.blocks[5]
        self.b7 = enet.blocks[6]

        self.after_blocks = nn.Sequential(
            enet.conv_head,
            enet.bn2,
            enet.act2
        )

        self.global_pool = enet.global_pool
        self.dropout = nn.Dropout(0.5)
        self.myfc = nn.Linear(enet.classifier.in_features, len(target_columns))
        # self.enet.classifier = nn.Identity()

        # ===============================
        # Mask netfork after b5
        # ===============================
        self.mask = nn.Sequential(
            nn.Conv2d(176, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )


    def forward(self, x, is_mask=True):
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        # =================================
        # Mask net
        # =================================
        mask = self.mask(x)

        x = self.b6(x)
        x = self.b7(x)

        x = self.after_blocks(x)
        x = self.global_pool(x)
        h = self.myfc(self.dropout(x))
        if is_mask:
            return h, mask
        else:
            return h

# =============================================================================
# Dataset
# =============================================================================

# Dual Cutout implementations
class CutoutV2(albumentations.DualTransform):
    def __init__(
        self,
        num_holes=8,
        max_h_size=8,
        max_w_size=8,
        fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        super(CutoutV2, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def apply(self, image, fill_value=0, holes=(), **params):
        return albumentations.functional.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")


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

        self.df_bbox = df.reset_index(drop=True)
        self.df = df.groupby("image_id").first().reset_index()
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
        bbox_df = self.df_bbox[self.df_bbox.image_id == row.image_id]

        if self.use_npy:
            # images = np.load(row.npy_path)
            images = cv2.imread(row.npy_path)
        else:
            images = pydicom.read_file(row.dicom_path).pixel_array

        # add bbox info into images
        bbox_mask = np.zeros((images.shape[0], images.shape[1]))
        if bbox_df.iloc[0]["have_box"]:
            for n_box in range(len(bbox_df)):
                box_info = bbox_df.iloc[n_box]
                bbox_mask[int(box_info["y"]):int(box_info["y"]+box_info["h"]),
                          int(box_info["x"]):int(box_info["x"]+box_info["w"])] = 1. # 255.
        # # add bbox mask into R of RGB
        # images[:,:,0] = bbox_mask

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
            aug = self.transform(image=images, mask=bbox_mask)                
            images_only = aug['image'].astype(np.float32).transpose(2, 0, 1) / 255                
            bbox_mask = aug['mask'].astype(np.float32)
            bbox_mask = cv2.resize(bbox_mask, (32, 32))[None, :, :]
        else:
            images = images.transpose(2, 0, 1)

        label = row[self.cols].values.astype(np.float16)

        # images_w_mask = torch.stack([
        #     # bbox_mask[0],
        #     images_only[1],
        #     images_only[2]
        # ])

        return {
            "image": torch.tensor(images_only, dtype=torch.float),
            # "image_w_mask": torch.tensor(images_w_mask, dtype=torch.float),
            "mask": torch.tensor(bbox_mask, dtype=torch.float),
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
        target_mask = data["mask"].to(device)

        bs = inputs.shape[0]

        with autocast(enabled=use_amp):
            output, output_mask = model(inputs)
            loss_cls = criterion(output, target).mean()
            loss_mask = criterion(output_mask, target_mask).mean()
            loss = loss_cls + loss_mask

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

        meters_dict["loss"].add(loss_cls.item(), n=bs)
        meters_dict["loss_mask"].add(loss_mask.item(), n=bs)
        meters_dict["loss_total"].add(loss.item(), n=bs)
        meters_dict["AP"].add(output=output.detach(), target=target)
        text_progress_bar = f"loss: {loss.item()} loss(avg): {meters_dict['loss_total'].value()[0]} " + \
            f"loss_cls(avg): {meters_dict['loss'].value()[0]} loss_mask(avg): {meters_dict['loss_mask'].value()[0]}"
        progress_bar.set_description(text_progress_bar)

    LOGGER.info(f"Train loss: {meters_dict['loss'].value()[0]}")
    LOGGER.info(f"Train loss_mask: {meters_dict['loss_mask'].value()[0]}")
    LOGGER.info(f"Train loss_total: {meters_dict['loss_total'].value()[0]}")
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
        target_mask = data["mask"].to(device)

        bs = inputs.shape[0]

        with autocast(enabled=use_amp):
            output, output_mask = model(inputs)
            loss_cls = criterion(output, target).mean()
            loss_mask = criterion(output_mask, target_mask).mean()
            loss = loss_cls + loss_mask

        meters_dict["loss"].add(loss_cls.item(), n=bs)
        meters_dict["loss_mask"].add(loss_mask.item(), n=bs)
        meters_dict["loss_total"].add(loss.item(), n=bs)
        meters_dict["AP"].add(output=output, target=target)
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {meters_dict['loss'].value()[0]}")

    LOGGER.info(f"Val loss: {meters_dict['loss'].value()[0]}")
    LOGGER.info(f"Val loss_mask: {meters_dict['loss_mask'].value()[0]}")
    LOGGER.info(f"Val loss_total: {meters_dict['loss_total'].value()[0]}")
    LOGGER.info(f"Val mAP: {meters_dict['AP'].value().mean()}")
    LOGGER.info(f"Val mAP score: {(2 * meters_dict['AP'].value().mean()) / 3}")
    LOGGER.info(f"Val time: {(time.time() - val_time) / 60:.3f} min")

    log_dict = {
        f"epoch": e,
        f"Loss/val_cv{cv}": meters_dict['loss'].value()[0],
        f"Loss_Mask/val_cv{cv}": meters_dict['loss_mask'].value()[0],
        f"Loss_Total/val_cv{cv}": meters_dict['loss_total'].value()[0],
        f"mAP/val_cv{cv}": meters_dict['AP'].value().mean(),
        f"mAP_metrics_old/val_cv{cv}": (2 * meters_dict['AP'].value()[:4].mean()) / 3
    }
 
    for n_t, t in enumerate(target_columns):
        log_dict[f"AP_{t}/val_cv{cv}"] = meters_dict['AP'].value()[n_t]
    wandb.log(log_dict)

    return meters_dict['AP'].value().mean()


def get_train_transforms(image_size):
    return albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomBrightness(limit=0.2, p=0.75),
    albumentations.RandomContrast(limit=0.2, p=0.75),

    albumentations.OneOf([
        albumentations.OpticalDistortion(distort_limit=1.),
        albumentations.GridDistortion(num_steps=5, distort_limit=1.),
    ], p=0.75),

    albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.75),
    CutoutV2(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.75),
    # ToTensorV2(p=1)
])


def get_val_transforms(image_size):
    return albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        # albumentations.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
        # ToTensorV2(p=1)
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

    use_bbox = cfg["use_bbox"]

    #######################################
    ## CV
    #######################################
    df = pd.read_csv(cfg["df_train_path"])
    if use_bbox == False:
        LOGGER.info(f"Drop bbox information")
        LOGGER.info(f"Group by image_id, and get first row")
        df = df.groupby("image_id").first().reset_index()

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

        optimizer = optim.AdamW(model.parameters(), lr=float(cfg["initial_lr"]), eps=1e-7)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=float(cfg["final_lr"]))
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, float(cfg["initial_lr"])) #(optimizer, T_max=n_epochs, eta_min=float(cfg["final_lr"]))

        # warmup_epo = 1
        # cosine_epo = n_epochs

        # optimizer = optim.Adam(model.parameters(), lr=float(cfg["initial_lr"]), eps=1e-7)
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
        # scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)


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
            "loss_mask": AverageValueMeter(),
            "loss_total": AverageValueMeter(),
            "AP": APMeter(),
        }

        for e in range(start_epoch , start_epoch + n_epochs):
            if e > 0:
                wandb.log({
                    "Learning Rate": optimizer.param_groups[0]["lr"],
                    "epoch": e
                })
                # scheduler_warmup.step(e-1)
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

