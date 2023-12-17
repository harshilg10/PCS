# testing
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt

print("1")
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
print("2")

import segmentation_models_pytorch as smp
print("Imported all libraries...")

model = smp.Unet(classes=1, in_channels=3, encoder_name="resnet34", encoder_weights="imagenet")
encoder = smp.encoders.get_encoder("resnet34", in_channels=3)

img = "gtCoarse/train/aachen/aachen_000000_000019_gtCoarse_color.png"
import cv2

img = cv2.imread(img)
x = cv2.resize(img, (256, 256))
x = np.transpose(x, (2, 0, 1))
encoded = encoder(torch.from_numpy(x).unsqueeze(0).float())

len(encoded)
