import matplotlib.pyplot as plt
import numpy as np
import os

def plot_sample_images(dataset, save_path, name=None, normalize=None, n_img=16):
    plt.figure(figsize=(16, 16))
    for i in range(n_img):
        data = dataset[i]
        if normalize == "+":
            img = data["image"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) \
                  + np.array([0.485, 0.456, 0.406])
        else:
            img = data["image"].numpy().transpose(1, 2, 0)

        img *= 255
        label = data["target"]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img.astype(np.uint8))
        plt.title(f"label: {label}")
    plt.tight_layout()
    if name:
        if ".png" in name:
            plt.savefig(os.path.join(save_path, f"{str(name)}.png"))
        else:
            plt.savefig(os.path.join(save_path, str(name)))
    else:
        plt.savefig(os.path.join(save_path, "sample_image.png"))
    plt.close()