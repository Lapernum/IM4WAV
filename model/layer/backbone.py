import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F

class Backbone:
    def __init__(self, size):
        self.BACKBONE_SIZE = size # in ("small", "base", "large" or "giant")

        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        
        self.backbone_arch = self.backbone_archs[self.BACKBONE_SIZE]
        self.backbone_name = f"dinov2_{self.backbone_arch}"
    
    def getBackbone(self):
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        # backbone_model.eval()
        backbone_model.cuda()
        return backbone_model