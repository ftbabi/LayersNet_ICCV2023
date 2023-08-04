from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
ACCURACY = MODELS
SIMULATORS = MODELS

def build_attention(cfg):
    return ATTENTION.build(cfg)

def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_simulator(cfg):
    return SIMULATORS.build(cfg)

def build_accuracy(cfg):
    return ACCURACY.build(cfg)
