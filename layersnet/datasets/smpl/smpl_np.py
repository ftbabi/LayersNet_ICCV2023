import numpy as np
import pickle
import os
import sys


# SMPL globals
SMPL_JOINT_NAMES = {
    0:  'Pelvis',
    1:  'L_Hip',        4:  'L_Knee',            7:  'L_Ankle',           10: 'L_Foot',
    2:  'R_Hip',        5:  'R_Knee',            8:  'R_Ankle',           11: 'R_Foot',
    3:  'Spine1',       6:  'Spine2',            9:  'Spine3',            12: 'Neck',            15: 'Head',
    13: 'L_Collar',     16: 'L_Shoulder',       18: 'L_Elbow',            20: 'L_Wrist',         22: 'L_Hand',
    14: 'R_Collar',     17: 'R_Shoulder',       19: 'R_Elbow',            21: 'R_Wrist',         23: 'R_Hand',
}
NUM_SMPL_JOINTS = len(SMPL_JOINT_NAMES)

class SMPLModel():
	def __init__(self, model_path, rest_pose=np.zeros(72)):
		"""
		SMPL model.

		Parameter:
		---------
		model_path: Path to the SMPL model parameters, pre-processed by
		`preprocess.py`.

		"""
		self.rest_pose = rest_pose.reshape((-1, 1, 3))
		with open(model_path, 'rb') as f:
			if sys.version_info[0] == 2: 
				params = pickle.load(f) # Python 2.x
			elif sys.version_info[0] == 3: 
				params = pickle.load(f, encoding='latin1') # Python 3.x
			self.J_regressor = params['J_regressor']
			self.weights = params['weights']
			self.posedirs = params['posedirs']
			self.v_template = params['v_template']
			self.shapedirs = params['shapedirs']
			self.faces = np.int32(params['f'])
			self.kintree_table = params['kintree_table']

		id_to_col = {
			self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
		}
		self.parent = {
			i: id_to_col[self.kintree_table[0, i]]
			for i in range(1, self.kintree_table.shape[1])
		}

		self.pose_shape = [24, 3]
		self.beta_shape = [10]
		self.trans_shape = [3]

		self.pose = np.zeros(self.pose_shape)
		self.beta = np.zeros(self.beta_shape)
		self.trans = np.zeros(self.trans_shape)

		self.verts = None
		self.J = None
		self.R = None

		# self.update()

	def set_params(self, pose=None, beta=None, trans=None, with_body=False, shape_verts=None):
		"""
		Set pose, shape, and/or translation parameters of SMPL model. Verices of the
		model will be updated and returned.

		Parameters:
		---------
		pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
		relative to parent joint. For root joint it's global orientation.
		Represented in a axis-angle format.

		beta: Parameter for model shape. A vector of shape [10]. Coefficients for
		PCA component. Only 10 components were released by MPI.

		trans: Global translation of shape [3].

		Return:
		------
		Updated vertices.

		"""
		if pose is not None:
			self.pose = pose
		if beta is not None:
			self.beta = beta
		if trans is not None:
			self.trans = trans
		# posed body
		G, root_offset = self.update(with_body, shape_verts=shape_verts)
		B = self.verts.copy() if with_body else None
		# rest pose body
		self.pose = self.rest_pose
		G_rest, _ = self.update(False, shape_verts=shape_verts)
		# from rest to pose
		for i in range(G.shape[0]):
			G[i] = G[i] @ np.linalg.inv(G_rest[i])
		return G, B, root_offset

	def update(self, with_body, shape_verts=None):
		"""
		Called automatically when parameters are updated.

		"""
		# how beta affect body shape
		if shape_verts is None:
			v_shaped = self.shapedirs.dot(self.beta) + self.v_template
		else:
			v_shaped = shape_verts
		# joints location
		self.J = self.J_regressor.dot(v_shaped)
		# align root joint with origin
		root_offset = self.J[:1].copy()
		v_shaped -= root_offset
		self.J -= root_offset
		pose_cube = self.pose.reshape((-1, 1, 3))
		# rotation matrix for each joint
		self.R = self.rodrigues(pose_cube)
		# world transformation of each joint
		G = np.empty((self.kintree_table.shape[1], 4, 4))
		G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
		for i in range(1, self.kintree_table.shape[1]):
			G[i] = G[self.parent[i]].dot(
				self.with_zeros(
					np.hstack(
						[self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
					)
				)
			)
		G = G - self.pack(
			np.matmul(
				G,
				np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
				)
			)

		if with_body:
			I_cube = np.broadcast_to(
				np.expand_dims(np.eye(3), axis=0),
				(self.R.shape[0]-1, 3, 3)
			)
			lrotmin = (self.R[1:] - I_cube).ravel()
			# how pose affect body shape in zero pose
			v_posed = v_shaped + self.posedirs.dot(lrotmin)	
			# transformation of each vertex
			T = np.tensordot(self.weights, G, axes=[[1], [0]])
			rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
			v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
			self.verts = v + self.trans.reshape([1, 3])
		
		return G, root_offset

	def rodrigues(self, r):
		"""
		Rodrigues' rotation formula that turns axis-angle vector into rotation
		matrix in a batch-ed manner.

		Parameter:
		----------
		r: Axis-angle rotation vector of shape [batch_size, 1, 3].

		Return:
		-------
		Rotation matrix of shape [batch_size, 3, 3].

		"""
		theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
		# avoid zero divide
		theta = np.maximum(theta, np.finfo(np.float64).eps)
		r_hat = r / theta
		cos = np.cos(theta)
		z_stick = np.zeros(theta.shape[0])
		m = np.dstack([
			z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
			r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
			-r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
		).reshape([-1, 3, 3])
		i_cube = np.broadcast_to(
			np.expand_dims(np.eye(3), axis=0),
			[theta.shape[0], 3, 3]
		)
		A = np.transpose(r_hat, axes=[0, 2, 1])
		B = r_hat
		dot = np.matmul(A, B)
		R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
		return R

	def with_zeros(self, x):
		"""
		Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

		Parameter:
		---------
		x: Matrix to be appended.

		Return:
		------
		Matrix after appending of shape [4,4]

		"""
		return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

	def pack(self, x):
		"""
		Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
		manner.

		Parameter:
		----------
		x: Matrices to be appended of shape [batch_size, 4, 1]

		Return:
		------
		Matrix of shape [batch_size, 4, 4] after appending.

		"""
		return np.dstack((np.zeros((x.shape[0], 4, 3)), x))