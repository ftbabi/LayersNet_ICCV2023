import torch
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import sys


def quads2tris(F):
	out = []
	for f in F:
		if len(f) == 3: out += [f]
		elif len(f) == 4: out += [[f[0], f[1], f[2]],
								  [f[0], f[2], f[3]]]
		else: sys.exit()
	return np.array(out, np.int32)

def faces2edges(F):
	E = set()
	for f in F:
		N = len(f)
		for i in range(N):
			j = (i + 1) % N
			E.add(tuple(sorted([f[i], f[j]])))
	E = list(E)
	return E

def edges2graph(E):
	G = {}
	for e in E:
		if not e[0] in G: G[e[0]] = {}
		if not e[1] in G: G[e[1]] = {}
		G[e[0]][e[1]] = 1
		G[e[1]][e[0]] = 1
	return G

def laplacianMatrix(F, with_diag=False):
	E = faces2edges(F)
	G = edges2graph(E)
	row, col, data = [], [], []
	for v in G:
		n = len(G[v])
		row += [v] * n
		col += [u for u in G[v]]
		data += [1.0 / n] * n
	if with_diag:
		row += [i for i in range(len(G))]
		col += [i for i in range(len(G))]
		data += [-1 for i in range(len(G))]
	return sparse.coo_matrix((data, (row, col)), shape=[len(G)] * 2)

def rotateByQuat(p, quat):
	R = np.zeros((3, 3))
	a, b, c, d = quat[3], quat[0], quat[1], quat[2]
	R[0, 0] = a**2 + b**2 - c**2 - d**2
	R[0, 1] = 2 * b * c - 2 * a * d
	R[0, 2] = 2 * b * d + 2 * a * c
	R[1, 0] = 2 * b * c + 2 * a * d
	R[1, 1] = a**2 - b**2 + c**2 - d**2
	R[1, 2] = 2 * c * d - 2 * a * b
	R[2, 0] = 2 * b * d - 2 * a * c
	R[2, 1] = 2 * c * d + 2 * a * b
	R[2, 2] = a**2 - b**2 - c**2 + d**2

	return np.dot(R, p)

def quatFromAxisAngle(axis, angle):
	axis /= np.linalg.norm(axis)

	half = angle * 0.5
	w = np.cos(half)

	sin_theta_over_two = np.sin(half)
	axis *= sin_theta_over_two

	quat = np.array([axis[0], axis[1], axis[2], w])

	return quat


def quatFromAxisAngle_var(axis, angle):
	axis /= torch.norm(axis)

	half = angle * 0.5
	w = torch.cos(half)

	sin_theta_over_two = torch.sin(half)
	axis *= sin_theta_over_two

	quat = torch.cat([axis, w])
	# print("quat size", quat.size())

	return quat

def patchwise(uv_groups, p_size=0.1, strict=True):
	verts_row, verts_col = [], []
	faces_row, faces_col = [], []

	for vg_name, uv_map in uv_groups.items():
		Vt_dict = uv_map['vertices'] # dict: idx=uv.co
		Ft = uv_map['faces'] # Original idx
		global_face_idx = uv_map.get('global_face_idx', None)
		modify_face_mapping = uv_map.get('modify_face_mapping', None)
		# Convert Ft to local Ft position
		Ft_loc = []
		for f in Ft:
			f_loc = []
			for i in f:
				uv_v = Vt_dict[i]
				f_loc.append(uv_v)
			f_loc = np.mean(np.stack(f_loc, axis=0), axis=0)
			Ft_loc.append(f_loc)
		Ft_loc = np.stack(Ft_loc, axis=0)

		v_row, v_col = [], []
		f_row, f_col = [], []

		# # Norm patch_co # if use island_pack when generating, this is no need
		# patch_co = (patch_co - np.min(patch_co, axis=0))/ (np.max(patch_co, axis=0) - np.min(patch_co, axis=0))
		# Split according to faces
		patch_co = np.floor(Ft_loc / p_size)
		# idx start from 0
		max_col = np.max(patch_co[:, 1]) + 1
		for i, ft in enumerate(Ft_loc):
			patch_idx = int(patch_co[i, 0] * max_col + patch_co[i, 1])
			# patch_idx = int(patch_co[i, 1])
			particle_idx = Ft[i]
			if not strict and global_face_idx[i] in modify_face_mapping.keys():
				# Deal with mapping for Cloth3D
				for v_mapping in modify_face_mapping[global_face_idx[i]]:
					for i in range(len(particle_idx)):
						if particle_idx[i] == v_mapping[1]:
							particle_idx[i] = v_mapping[0]
							break

			v_row.extend(particle_idx)
			v_col.extend([patch_idx] * len(particle_idx))
			f_row.append(i)
			f_col.append(patch_idx)
		# Clean patch idx, delete empty patch
		sorted_fcol = sorted(list(set(f_col)))
		clean_col_idx = {
			col_idx: order_idx
			for order_idx, col_idx in enumerate(sorted_fcol)
		}
		assert len(clean_col_idx) <= len(Ft)
		f_col = [clean_col_idx[i] for i in f_col]
		v_col = [clean_col_idx[i] for i in v_col]

		verts_rc = list(set(zip(v_row, v_col)))
		verts_row.append(np.array([i[0] for i in verts_rc]))
		verts_col.append(np.array([i[1] for i in verts_rc]))
		faces_row.append(np.array(f_row))
		faces_col.append(np.array(f_col))
	
	# Move offset
	patch_offset = [0]
	for i in range(len(verts_col)):
		col_len = np.max(verts_col[i]) + 1
		patch_offset.append(col_len)
	patch_offset = np.cumsum(patch_offset)

	for i in range(len(verts_row)):
		verts_col[i] += patch_offset[i]
		faces_col[i] += patch_offset[i]
	
	verts_row = np.concatenate(verts_row, axis=0)
	verts_col = np.concatenate(verts_col, axis=0)
			
	return verts_row, verts_col, faces_row, faces_col, patch_offset
