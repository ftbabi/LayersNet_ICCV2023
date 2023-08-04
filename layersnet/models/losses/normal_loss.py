import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from layersnet.utils import face_normals_batched


def normal_error(pred, label, weight=None, reduction='sum', avg_factor=None, indices_weight=None, faces=None, v2f_mask_sparse=None, eps=1e-7, obj_wise=True, **kwargs):
    assert faces is not None and v2f_mask_sparse is not None
    faces = faces.transpose(-1, -2)
    v2f_mask_sparse = torch.sparse_coo_tensor(**v2f_mask_sparse)

    bs, n_v, n_f = v2f_mask_sparse.shape
    pred_face_normals = face_normals_batched(pred, faces)
    pred_vert_normals = torch.bmm(v2f_mask_sparse, pred_face_normals)
    pred_vert_normals_normed = pred_vert_normals / (torch.norm(pred_vert_normals, dim=-1, keepdim=True) + eps)

    gt_face_normals = face_normals_batched(label, faces)
    gt_vert_normals = torch.bmm(v2f_mask_sparse, gt_face_normals)
    gt_vert_normals_normed = gt_vert_normals / (torch.norm(gt_vert_normals, dim=-1, keepdim=True) + eps)
    
    # element-wise losses
    loss = F.mse_loss(pred_vert_normals_normed, gt_vert_normals_normed, reduction='none')
    loss = torch.sum(loss, dim=-1)

    if indices_weight is not None:
        if len(indices_weight.shape) > 2:
            indices_weight = indices_weight.squeeze(1)
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
class NormalLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_normal',
                 obj_wise=True,
                 ):
        super(NormalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.obj_wise = obj_wise

        self.criterion = normal_error
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
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            obj_wise=self.obj_wise,
            **kwargs)
        loss *= self.loss_weight
        return loss

    @property
    def loss_name(self):
        return self._loss_name
