import urllib

import mmcv
from mmcv.runner import load_checkpoint

import model.dinov2.eval.segmentation_m2f.models.segmentors

from backbone import *
from utils import *

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

def getSegmentationHead():
    HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
    HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
    HEAD_TYPE = "ms" # in ("ms, "linear")

    backbone_cls = Backbone("small")
    backbone_model = backbone_cls.getBackbone()

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_cls.backbone_name}/{backbone_cls.backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_cls.backbone_name}/{backbone_cls.backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    if HEAD_TYPE == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

    model = create_segmenter(cfg, backbone_model=backbone_model)
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.cuda()

    return model

def getSegmentationModel():
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"

    CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
    CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

    cfg_str = load_config_from_url(CONFIG_URL)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    model = init_segmentor(cfg)
    load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
    model.cuda()

    return model