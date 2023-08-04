import torch
import torch.nn as nn

from collections import defaultdict

from ..builder import HEADS, build_loss, build_accuracy
from .base_head import BaseHead


@HEADS.register_module()
class SimHead(BaseHead):
    def __init__(self,
                 init_cfg=None,
                 loss_decode=dict(type='L2Loss', reduction='sum', loss_weight=1.0),
                 accuracy=dict(type='L2Accuracy', reduction='mean'),
                 *args,
                 **kwargs):
        super(SimHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        # LOSS
        if isinstance(loss_decode, dict):
            loss_decode = [loss_decode]
        elif isinstance(loss_decode, (list, tuple)):
            pass
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        self.loss_decode = nn.ModuleList()
        for loss in loss_decode:
            self.loss_decode.append(build_loss(loss))
        
        # ACCURACY
        if isinstance(accuracy, dict):
            accuracy = [accuracy]
        elif isinstance(accuracy, (list, tuple)):
            pass
        elif accuracy is None:
            accuracy = []
        else:
            raise TypeError(f'accuracy must be a dict or sequence of dict,\
                but got {type(accuracy)}')
        self.accuracy_decode = nn.ModuleList()
        for accs in accuracy:
            self.accuracy_decode.append(build_accuracy(accs))

    def loss(self, pred, gt_label, reduction_override=None, weight=None, indices=None, indices_type=None, **kwargs):
        losses = dict()

        # compute loss
        for loss_decode in self.loss_decode:
            loss = loss_decode(
                pred,
                gt_label,
                weight=weight,
                reduction_override=reduction_override,
                indices=indices,
                **kwargs)
            if loss_decode.loss_name not in losses:
                losses[loss_decode.loss_name] = loss
            else:
                losses[loss_decode.loss_name] += loss

        # compute accuracy
        acc_dict = self.accuracy(pred, gt_label, reduction_override=reduction_override, indices=indices, indices_type=indices_type, **kwargs)
        losses.update(acc_dict)

        return losses

    def accuracy(self, pred, gt_label, reduction_override=None, indices=None, indices_type=None, **kwargs):
        acc_dict = defaultdict(list)
        # compute accuracy
        for accuracy_decode in self.accuracy_decode:
            accs = accuracy_decode(
                pred,
                gt_label,
                indices=indices,
                indices_type=indices_type,
                reduction_override=reduction_override,
                **kwargs)
            for key, val in accs.items():
                acc_dict[key].append(val)
        rst = dict()
        for key, val in acc_dict.items():
            rst[key] = torch.mean(torch.stack(val))
        return rst

    def forward_train(self, pred, gt_label):
        losses = self.loss(pred, gt_label)
        return losses, pred
