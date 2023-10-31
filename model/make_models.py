import torch

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import json
import glob
from tqdm.notebook import tqdm
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vitl14.to(device)​
transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])