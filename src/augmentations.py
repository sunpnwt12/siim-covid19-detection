from lib.include import *
from CONFIG import GlobalConfig

cfg = GlobalConfig()

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5, shift_limit=0, rotate_limit=6),
        A.RandomResizedCrop(p=0.5, height=cfg.image_size, width=cfg.image_size),
        A.Cutout(num_holes=10, max_h_size=64, max_w_size=64, p=0.5),
        A.RandomGamma(p=0.5, gamma_limit=(95, 105)),
        A.RandomBrightnessContrast(p=0.5),
        A.Resize(cfg.image_size, cfg.image_size),
        ToTensorV2()
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(cfg.image_size, cfg.image_size),
        ToTensorV2()
    ])