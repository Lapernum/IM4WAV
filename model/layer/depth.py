import urllib
import torch
import torch.nn as nn

import mmcv
from mmcv.runner import load_checkpoint

from backbone import *
from utils import *


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

def getDepthModel():
    HEAD_DATASET = "nyu" # in ("nyu", "kitti")
    HEAD_TYPE = "dpt" # in ("linear", "linear4", "dpt")
    
    backbone_cls = Backbone("small")
    backbone_model = backbone_cls.getBackbone()

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_cls.backbone_name}/{backbone_cls.backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_cls.backbone_name}/{backbone_cls.backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    model = create_depther(
        cfg,
        backbone_model=backbone_model,
        backbone_size=backbone_cls.BACKBONE_SIZE,
        head_type=HEAD_TYPE,
    )

    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    # model.eval()
    model.cuda()
    return model