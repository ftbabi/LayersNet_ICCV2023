import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def l2_error(pred, label, weight=None, reduction='sum', avg_factor=None, indices_weight=None, obj_wise=True, **kwargs):
    loss = F.mse_loss(pred, label, reduction='none')
    loss = torch.sum(loss, dim=-1)
    if indices_weight is not None:
        if len(indices_weight.shape) > 2:
            indices_weight = indices_weight.squeeze(1)
        # indices_weight should be 1/N, while others are 0
        if not obj_wise:
            indices_weight = indices_weight > 0
            indices_weight = indices_weight / torch.sum(indices_weight, dim=-1, keepdim=True)
        loss = loss * indices_weight

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    assert not torch.isnan(loss)

    return loss


@LOSSES.register_module()
class L2Loss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_l2',
                 obj_wise=True,):
        super(L2Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.obj_wise = obj_wise

        self.criterion = l2_error
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            obj_wise=self.obj_wise,
            **kwargs)
        loss *= self.loss_weight
        return loss

    @property
    def loss_name(self):
        return self._loss_name
