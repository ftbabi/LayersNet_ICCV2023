# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base head."""

    def __init__(self, init_cfg=None):
        super(BaseHead, self).__init__(init_cfg)

    @abstractmethod
    def forward_train(self, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, **kwargs):
        pass