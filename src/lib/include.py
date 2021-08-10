import ast
import gc
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import wandb
import hydra
from omegaconf import OmegaConf
from albumentations.augmentations.bbox_utils import (convert_bbox_from_albumentations, convert_bbox_to_albumentations)
from albumentations.pytorch.transforms import ToTensorV2
from madgrad import MADGRAD
from pydicom.pixel_data_handlers.util import apply_voi_lut
from termcolor import colored
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.efficientnet import *
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import (DataLoader, Dataset, RandomSampler, SequentialSampler)
from torchmetrics import AUROC, Accuracy, AveragePrecision 
from tqdm.notebook import tqdm

# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import average_precision_score

