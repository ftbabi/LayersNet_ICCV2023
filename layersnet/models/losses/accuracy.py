# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ACCURACY
from layersnet.datasets.utils import GARMENT_TYPE
from layersnet.utils import face_normals_batched
from mmcv.ops import QueryAndGroup


def accuracy_l2(pred, label, indices, indices_type, prefix='', **kwargs):
    acc_dict = defaultdict(list)
    # element-wise losses
    square_error = F.mse_loss(pred, label, reduction='none')
    square_error = torch.sqrt(torch.sum(square_error, dim=-1))


    # Calculate per outfit error
    for b_error, b_ind, b_type in zip(square_error, indices, indices_type):
        for i in range(1, b_ind.shape[0]):
            g_type_idx = torch.argmax(b_type[i-1], dim=0)
            g_type = GARMENT_TYPE[g_type_idx]
            start, end = b_ind[i-1], b_ind[i]
            g_error = torch.mean(b_error[start:end])
            acc_dict[g_type].append(g_error)
    
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }
    # Align the length of keys
    for g_type in GARMENT_TYPE:
        key = f"{prefix}.{g_type}"
        if key not in acc_dict.keys():
            acc_dict[key] = torch.zeros(1).cuda()

    return acc_dict

def accuracy_collision_count(query_mesh, anchor_mesh, anchor_normals, grouper, max_dist=5.0, **kwargs):
    anchor_mesh = anchor_mesh.contiguous()
    query_mesh = query_mesh.contiguous()
    grouped_results = grouper(anchor_mesh, query_mesh, anchor_normals)
    grouped_normals, grouped_xyz = grouped_results
    grouped_diff = query_mesh.transpose(1, 2).unsqueeze(-1) - grouped_xyz  # relative offsets
    grouped_normals = grouped_normals.permute(0, 2, 3, 1)
    grouped_diff = grouped_diff.permute(0, 2, 3, 1)
    grouped_diff_l2 = torch.sqrt(torch.sum(grouped_diff**2, dim=-1))

    dot = torch.sum(grouped_diff * grouped_normals, dim=-1)
    valid_mask = grouped_diff_l2 <= max_dist
    dot *= valid_mask
    collision_mask = dot < 0
    assert collision_mask.shape[-1] == 1
    collision_mask = collision_mask.squeeze(-1)
    num_collisions = torch.sum(collision_mask, dim=-1, keepdim=True)

    return num_collisions

def accuracy_collision(garment_mesh, indices, indices_type, faces, v2f_mask_sparse, h_state, h_faces, h_v2f_mask_sparse, grouper, prefix='', eps=1e-7, **kwargs):
    acc_dict = defaultdict(list)
    bs = garment_mesh.shape[0]
    faces = faces.transpose(-1, -2)
    h_faces = h_faces.transpose(-1, -2)
    h_v2f_mask_sparse = torch.sparse_coo_tensor(**h_v2f_mask_sparse)
    v2f_mask_sparse = torch.sparse_coo_tensor(**v2f_mask_sparse)

    garment_face_normals = face_normals_batched(garment_mesh, faces)
    garment_vert_normals = torch.bmm(v2f_mask_sparse, garment_face_normals)
    garment_vert_normals_normed = garment_vert_normals / (torch.norm(garment_vert_normals, dim=-1, keepdim=True) + eps)
    garment_vert_normals_normed = garment_vert_normals_normed.transpose(-1, -2).contiguous()

    human_face_normals = face_normals_batched(h_state, h_faces)
    human_vert_normals = torch.bmm(h_v2f_mask_sparse, human_face_normals) / (torch.bmm(h_v2f_mask_sparse, torch.ones(bs, h_v2f_mask_sparse.shape[-1], 1).cuda()) + eps)
    human_vert_normals_normed = human_vert_normals / (torch.norm(human_vert_normals, dim=-1, keepdim=True) + eps)
    human_vert_normals_normed = human_vert_normals_normed.transpose(-1, -2).contiguous()

    # Gamrent to human
    g2h_collision_count = accuracy_collision_count(garment_mesh, h_state, human_vert_normals_normed, grouper)
    garment_verts_num = torch.tensor([i[-1] for i in indices]).reshape(garment_mesh.shape[0], -1).cuda()
    g2h_collision_rate = g2h_collision_count / garment_verts_num
    acc_dict['garment2human'].append(g2h_collision_rate)
    
    # Garment layers
    for g_mesh, g_normals, b_ind, b_type in zip(garment_mesh, garment_vert_normals_normed, indices, indices_type):
        outer_mesh = g_mesh[b_ind[-2]:b_ind[-1]].unsqueeze(0)
        inner_mesh = g_mesh[:b_ind[-2]].unsqueeze(0)
        inner_normals =g_normals[:, :b_ind[-2]].unsqueeze(0)
        outer_collision_count = accuracy_collision_count(outer_mesh, inner_mesh, inner_normals, grouper)
        outer_collision_rate = outer_collision_count / (b_ind[-1] - b_ind[-2])
        outer_type_idx = torch.argmax(b_type[-1], dim=0)
        outer_type = GARMENT_TYPE[outer_type_idx]
        acc_dict[outer_type].append(outer_collision_rate)

    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }
    # Align the length of keys
    for g_type in GARMENT_TYPE:
        key = f"{prefix}.{g_type}"
        if key not in acc_dict.keys():
            acc_dict[key] = torch.tensor(-1.).cuda()

    return acc_dict

@ACCURACY.register_module()
class L2Accuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_l2'):
        super(L2Accuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name

    def forward(self, pred, target, indices, indices_type=None, **kwargs):
        return accuracy_l2(pred, target, indices, indices_type=indices_type, prefix=self.acc_name, **kwargs)
    
    @property
    def acc_name(self):
        return self._acc_name

@ACCURACY.register_module()
class CollisionAccuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_collision'):
        super(CollisionAccuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name
        self.grouper = QueryAndGroup(
            max_radius=None,
            sample_num=1,
            min_radius=0,
            use_xyz=False,
            normalize_xyz=False,
            return_grouped_xyz=True,
            return_grouped_idx=False,
            return_unique_cnt=False,
        )

    def forward(self, pred, target, indices, indices_type=None, **kwargs):
        return accuracy_collision(pred, indices, indices_type=indices_type, grouper=self.grouper, prefix=self.acc_name, **kwargs)
    
    @property
    def acc_name(self):
        return self._acc_name