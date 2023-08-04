import torch
import torch.nn as nn
from mmcv.ops import QueryAndGroup

from ..builder import LOSSES
from .utils import weight_reduce_loss
from layersnet.utils import face_normals_batched


def garment_wise_collision_error(pred, label, faces, v2f_mask_sparse, thresh, max_dist, weight=None, reduction='sum', avg_factor=None, grouper=None, indices=None, eps=1e-7,**kwargs):
    v2f_mask_sparse = torch.sparse_coo_tensor(**v2f_mask_sparse)
    faces = faces.transpose(-1, -2)
    bs, n_v, n_f = v2f_mask_sparse.shape
    
    g_face_normals = face_normals_batched(pred, faces)
    g_vert_normals = torch.bmm(v2f_mask_sparse, g_face_normals)
    g_vert_normals = g_vert_normals / (torch.norm(g_vert_normals, dim=-1, keepdim=True) + eps)
    g_vert_normals = g_vert_normals.transpose(-1, -2)

    batch_loss = 0

    for g_mesh, g_normals, b_ind in zip(pred, g_vert_normals, indices):
        outer_mesh = g_mesh[b_ind[-2]:b_ind[-1]].unsqueeze(0).contiguous()
        inner_mesh = g_mesh[:b_ind[-2]].unsqueeze(0).contiguous()
        inner_normals =g_normals[:, :b_ind[-2]].unsqueeze(0).contiguous()
        grouped_results = grouper(inner_mesh, outer_mesh, inner_normals)

        grouped_normals, grouped_xyz = grouped_results
        grouped_diff = outer_mesh.transpose(1, 2).unsqueeze(-1) - grouped_xyz  # relative offsets
        grouped_normals = grouped_normals.permute(0, 2, 3, 1)
        grouped_diff = grouped_diff.permute(0, 2, 3, 1)

        dot = torch.sum(grouped_diff * grouped_normals, dim=-1)
        valid_dot = dot - thresh
        if max_dist is not None:
            valid_mask = dot >= -max_dist
            valid_mask.requires_grad = False
            # min(n * d - e, 0)
            valid_dot = valid_dot * valid_mask
        mask = valid_dot < 0
        mask.requires_grad = False
        # Get the mean loss
        loss = torch.sum(valid_dot * mask / (torch.sum(mask, dim=-1, keepdim=True) + eps), dim=-1) ** 2

        avg_mask = loss > 0
        avg_mask.requires_grad = False

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        loss /= (torch.sum(avg_mask) + eps)
        batch_loss += loss
    
    return batch_loss

def collision_error(pred, label, h_state, h_faces, h_v2f_mask_sparse, thresh, max_dist, weight=None, reduction='sum', avg_factor=None, grouper=None, eps=1e-7, **kwargs):
    h_v2f_mask_sparse = torch.sparse_coo_tensor(**h_v2f_mask_sparse)
    human_faces = h_faces.transpose(-1, -2)
    human_verts = h_state.transpose(-1, -2)[..., :3].contiguous()
    bs, n_v, n_f = h_v2f_mask_sparse.shape

    pred = pred.contiguous()
    
    human_face_normals = face_normals_batched(human_verts, human_faces)
    human_vert_normals = torch.bmm(h_v2f_mask_sparse, human_face_normals) / (torch.bmm(h_v2f_mask_sparse, torch.ones(bs, h_v2f_mask_sparse.shape[-1], 1).cuda()) + eps)
    human_vert_normals = human_vert_normals.transpose(-1, -2).contiguous()

    grouped_results = grouper(human_verts, pred, human_vert_normals)

    grouped_normals, grouped_xyz = grouped_results
    grouped_diff = pred.transpose(1, 2).unsqueeze(-1) - grouped_xyz  # relative offsets
    grouped_normals = grouped_normals.permute(0, 2, 3, 1)
    grouped_diff = grouped_diff.permute(0, 2, 3, 1)
    grouped_normals = grouped_normals / (torch.norm(grouped_normals, dim=-1, keepdim=True) + eps)

    dot = torch.sum(grouped_diff * grouped_normals, dim=-1)
    valid_dot = dot - thresh
    if max_dist is not None:
        valid_mask = dot >= -max_dist
        valid_mask.requires_grad = False
        # min(n * d - e, 0)
        valid_dot = valid_dot * valid_mask
    mask = valid_dot < 0
    mask.requires_grad = False
    # Get the mean loss
    loss = torch.sum(valid_dot * mask / (torch.sum(mask, dim=-1, keepdim=True) + eps), dim=-1) ** 2

    avg_mask = loss > 0
    avg_mask.requires_grad = False

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    loss /= (torch.sum(avg_mask) + eps)
    
    return loss


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    def __init__(self,
                 reduction='sum',
                 loss_weight=1.0,
                 per_garment=False,
                 per_garment_weight=1.0,
                 loss_name='loss_collision',
                 # This is aligned with the dataset collision distance
                 thresh=0.001,
                 radius=0.01,
                 min_radius=0.0,
                 sample_num=4,
                 use_xyz=False,
                 normalize_xyz=False,
                 return_grouped_xyz=True,
                 return_grouped_idx=False,
                 return_unique_cnt=False,):
        super(CollisionLoss, self).__init__()
        self.grouper = QueryAndGroup(
            radius,
            sample_num,
            min_radius=min_radius,
            use_xyz=use_xyz,
            normalize_xyz=normalize_xyz,
            return_grouped_xyz=return_grouped_xyz,
            return_grouped_idx=return_grouped_idx,
            return_unique_cnt=return_unique_cnt,
        )
        self.per_garment = per_garment
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.per_garment_weight = per_garment_weight
        self.thresh = thresh
        self.radius = radius

        self.criterion = collision_error
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
        if self.loss_weight == 0:
            return 0
        else:
            loss_cls = self.loss_weight * self.criterion(
                cls_score,
                label,
                weight=weight,
                reduction=reduction,
                avg_factor=avg_factor,
                thresh=self.thresh,
                grouper=self.grouper,
                max_dist=self.radius,
                **kwargs)
            if self.per_garment:
                loss_pg = self.per_garment_weight * garment_wise_collision_error(
                    cls_score,
                    label,
                    weight=weight,
                    reduction=reduction,
                    avg_factor=avg_factor,
                    thresh=self.thresh,
                    grouper=self.grouper,
                    max_dist=self.radius,
                    **kwargs)
                loss_cls += loss_pg
            return loss_cls

    @property
    def loss_name(self):
        return self._loss_name
