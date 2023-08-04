import torch
from torch import nn as nn
import scipy.sparse as sp
from scipy.sparse import vstack, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as R
import numpy as np

from mmcv.ops import grouping_operation
# from layersnet.datasets.utils import to_numpy_detach


def face_normals_batched(verts, faces, eps=1e-7):
	face_verts = grouping_operation(verts.transpose(-1, -2), faces.int()).permute(0, 2, 3, 1)
	face_normals = torch.cross(
		face_verts[:, :, 2] - face_verts[:, :, 1],
		face_verts[:, :, 0] - face_verts[:, :, 1],
		dim=-1,
	)

	face_normals = face_normals / (torch.norm(face_normals, dim=-1, keepdim=True) + eps)
	return face_normals

def rotation_from_normals(data, eps=1e-7):
	bs, n_verts, dim = data.shape
	random_x = torch.zeros_like(data).to(data)
	random_x[:, :, 0] = 1
	zero_position = torch.where(torch.abs(data) < eps)
	random_x[zero_position] = 1

	# Calculate orthogonal vector
	orth_x = random_x - torch.sum(data * random_x, dim=-1, keepdim=True) * data
	orth_x = orth_x / (torch.linalg.norm(orth_x, dim=-1, ord=2, keepdim=True) + eps)

	orth_y = torch.cross(data, orth_x)
	rot_M = torch.stack([orth_x, orth_y, data], dim=-1)
	return rot_M.transpose(-1, -2)

def rotation_from_quats_np(data, eps=1e-7):
	seq_len, n_history, dim = data.shape
	data = data.reshape(-1, dim)
	# Wind q: wxyz format
	# Func need xyzw
	data_q = np.zeros_like(data)
	w = data[:, 0]
	data_q[:, 0:3] = data[:, 1:4]
	data_q[:, 3] = w
	sum_q = np.sum(data_q == 0, axis=-1)
	zero_q = np.where(sum_q == dim)
	data_q[zero_q, 3] = 1
	rot_q = R.from_quat(data_q).as_matrix()
	rot_dim = rot_q.shape[-1]
	rot_q = rot_q.reshape(seq_len, n_history, rot_dim, rot_dim)
	
	return rot_q


def extract_rotation_with_padding(rotation_M, group_idx, mask=None, pad_dim=3):
	bs = rotation_M.shape[0]
	src_num = rotation_M.shape[1]
	tar_num = group_idx.shape[1]

	grouped_rotation = grouping_operation(rotation_M.reshape(bs, src_num, -1).transpose(-1, -2), group_idx).permute(0, 2, 3, 1).reshape(bs, tar_num, -1, 3, 3)
	assert grouped_rotation.shape[2] == 1
	grouped_rotation = grouped_rotation.squeeze(2)
	if mask is not None:
		grouped_rotation *= mask.unsqueeze(-1)
		grouped_rotation[torch.where(mask.squeeze(-1) == False)] = torch.eye(pad_dim).to(grouped_rotation)
	
	return grouped_rotation