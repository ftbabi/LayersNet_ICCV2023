import os
import torch
import numpy as np
from mmcv.parallel import collate, DataContainer
from collections import defaultdict
from itertools import chain

from layersnet.datasets.utils.mesh import laplacianMatrix
from .builder import DATASETS
from layersnet.datasets.utils import SparseMask, faces2edges, edges2graph
from layersnet.datasets.utils import collate_sparse, normalize
from layersnet.datasets.utils import readH5
from .layers_dynamic_dataset import LayersDataset
from layersnet.utils.mesh3d import rotation_from_quats_np


@DATASETS.register_module()
class PatchLayersDataset(LayersDataset):

    def __init__(self, env_cfg, phase, **kwargs):
        super(PatchLayersDataset, self).__init__(env_cfg=env_cfg, phase=phase, **kwargs)

    def prepare_rollout(self, batch_data, current_garment, **kwargs):
        batch_data['inputs']['state'] = DataContainer(current_garment.transpose(-1, -2), stack=True, padding_value=0, pad_dims=1)
        return batch_data

    def _process_wind(self, wind, global_trans, trans_mean_std):
        wind_attr = global_trans.reshape(self.env_cfg.history + 1, -1)
        wind_attr = normalize([wind_attr], [trans_mean_std])[0]
        assert wind_attr.shape[-1] == 9
        wind_attr = wind_attr[:, 3:] # first 3 is pos, which is no use; only need vel and acc
        wind = wind.reshape(self.env_cfg.history + 1, -1)
        wind = self.clothenv_reader.normalize_wind(wind)
        wind = np.concatenate([wind_attr, wind], axis=-1)
        wind = wind.reshape(1, -1)

        return wind

    def load_raw_dynamic(self, data_names, h5_path, omit_key=None, **kwargs):
        g_types, names, offset, pos, vel, acc, wind = readH5(data_names, h5_path)

        # Omit specific clothes, e.g. jacket
        if omit_key is not None:
            vertices_num = [offset[i] - offset[i-1] for i in range(1, offset.shape[0])]
            filtered_g_types = []
            filtered_names = []
            offset_span = []
            selected_vertices = [0]
            string_names = names.astype(str)
            assert names.shape[0] + 1 == offset.shape[0]
            for idx, tar_name in enumerate(string_names):
                if tar_name in omit_key:
                    # Clear g_types
                    continue
                if idx < g_types.shape[0]:
                    filtered_g_types.append(g_types[idx])
                filtered_names.append(names[idx])
                offset_span.append([offset[idx], offset[idx+1]])
                selected_vertices.append(vertices_num[idx])
            f_g_types = np.array(filtered_g_types)
            f_names = np.array(filtered_names)
            f_pos = np.concatenate([pos[span[0]:span[1]] for span in offset_span], axis=0)
            f_vel = np.concatenate([vel[span[0]:span[1]] for span in offset_span], axis=0)
            f_acc = np.concatenate([acc[span[0]:span[1]] for span in offset_span], axis=0)
            f_offset = np.cumsum(selected_vertices)

            g_types = f_g_types
            names = f_names
            pos = f_pos
            vel = f_vel
            acc = f_acc
            offset = f_offset

        garment_offset = offset[0:-2]
        human_offset = offset[-3:-1]
        trans_offset = offset[-2:]

        # Concat needed info
        state_list = []
        if self.env_cfg.get('with_pos', True):
            state_list.append(pos)
        if self.env_cfg.get('with_vel', True):
            state_list.append(vel)
        state_list = np.concatenate(state_list, axis=-1)

        wind = wind.reshape(1, -1)
        trans_state = np.concatenate([
            pos[trans_offset[0]:trans_offset[1]],
            vel[trans_offset[0]:trans_offset[1]],
            acc[trans_offset[0]:trans_offset[1]]], axis=-1)
        input_states = dict(states=state_list, wind=wind, trans_state=trans_state)

        return g_types, names, offset, input_states

    def load_raw_static(self, data_names, h5_path, omit_key=None):
        mesh_name, mesh_offset, mesh_face_offset, mesh_faces = readH5(data_names, h5_path)
        if omit_key is not None:
            vertices_num = [mesh_offset[i] - mesh_offset[i-1] for i in range(1, mesh_offset.shape[0])]
            faces_num = [mesh_face_offset[i] - mesh_face_offset[i-1] for i in range(1, mesh_face_offset.shape[0])]
            filtered_mesh_name = []
            selected_vertices = [0]
            selected_faces = [0]
            original_faces = []
            for i in range(1, mesh_face_offset.shape[0]):
                cur_faces = mesh_faces[mesh_face_offset[i-1]:mesh_face_offset[i]]
                cur_faces -= mesh_offset[i-1]
                assert np.min(cur_faces) == 0 and np.max(cur_faces) + 1 == vertices_num[i-1]
                original_faces.append(cur_faces)
            filtered_faces = []
            string_names = mesh_name.astype(str)
            assert mesh_name.shape[0] + 1 == mesh_offset.shape[0]
            for idx, tar_name in enumerate(string_names):
                if tar_name in omit_key:
                    # Clear g_types
                    continue
                filtered_mesh_name.append(mesh_name[idx])
                filtered_faces.append(original_faces[idx])
                selected_vertices.append(vertices_num[idx])
                selected_faces.append(faces_num[idx])
            f_mesh_name = np.array(filtered_mesh_name)
            f_mesh_offset = np.cumsum(selected_vertices)
            f_mesh_face_offset = np.cumsum(selected_faces)
            for i in range(len(filtered_faces)):
                filtered_faces[i] += f_mesh_offset[i]
            filtered_faces = np.concatenate(filtered_faces, axis=0)
            
            mesh_name = f_mesh_name
            mesh_offset = f_mesh_offset
            mesh_face_offset = f_mesh_face_offset
            mesh_faces = filtered_faces

        return mesh_name, mesh_offset, mesh_face_offset, mesh_faces

    def __getitem__(self, idx):
        # Remember this order: garment, human, attr, invisable, patches
        g2p_mask = SparseMask(prefix='g2p')
        gmesh_mask = SparseMask(prefix='gmesh')
        pmesh_mask = SparseMask(prefix='pmesh')
        attr_mask = SparseMask(prefix='attr')
        invisable_force_mask = SparseMask(prefix='invisable_force')

        seq_num, frame_idx = self.data_list[idx]

        non_grad_step = self.non_grad_step

        # Load global info
        mean_std = self.stat[:, :-3]
        trans_mean_std = self.trans_stat

        frame_path = os.path.join(self.clothenv_reader.generated_dir, seq_num, "rollout", f"{frame_idx}.h5")
        ## Behavior of input_states: 1. stack with history
        g_types, names, offset, input_states = self.load_raw_dynamic(self.data_names, frame_path, self.env_cfg.get('omit_keys', None))
        garment_offset = offset[0:-2]
        human_offset = offset[-3:-1]
        trans_offset = offset[-2:]
        per_frame_states_dim = dict()
        for key, val in input_states.items():
            per_frame_states_dim[key] = val.shape[-1]
        # History info
        history_input_states = defaultdict(list)
        for key, val in input_states.items():
            history_input_states[key].append(val)
        
        for i in range(self.env_cfg.history):
            # Concat history info
            his_frame_idx = frame_idx - (1+i)
            his_frame_path = os.path.join(self.clothenv_reader.generated_dir, seq_num, "rollout", f"{his_frame_idx}.h5")
            if not os.path.exists(his_frame_path):
                continue
            _, _, _, his_input_states = self.load_raw_dynamic(self.data_names, his_frame_path, self.env_cfg.get('omit_keys', None))
            for key, val in his_input_states.items():
                assert key in history_input_states.keys()
                history_input_states[key].append(val)
        
        # Padding the init one
        for key, val in history_input_states.items():
            earliest_his_states = val[-1]
            # this will pad only when current length is less than the history
            pad_dim = self.env_cfg.history - len(val)
            for i in range(pad_dim):
                history_input_states[key].append(earliest_his_states)

        # Concat the history
        states_list = dict()
        for key, val in history_input_states.items():
            cat_val = np.concatenate(val, axis=-1)
            states_list[key] = [cat_val]

        # Load non grad step
        for i in range(non_grad_step):           
            next_frame_idx = frame_idx + i+1
            frame_path = os.path.join(self.clothenv_reader.generated_dir, seq_num, "rollout", f"{next_frame_idx}.h5")
            _g_types, _names, _offset, _input_states = self.load_raw_dynamic(self.data_names, frame_path, self.env_cfg.get('omit_keys', None))
            for key, val in _input_states.items():
                # per_frame_states_dim = val.shape[-1]
                step_states_his = np.concatenate([val, states_list[key][-1][:, :-per_frame_states_dim[key]]], axis=-1)
                states_list[key].append(step_states_his)
        
        # Load GT
        _, _, _, gt_states = self.load_raw_dynamic(
            self.data_names,
            os.path.join(self.clothenv_reader.generated_dir, seq_num, "rollout", f"{frame_idx + non_grad_step+1}.h5"),
            self.env_cfg.get('omit_keys', None))
        garment_gt = gt_states['states'][garment_offset[0]:garment_offset[-1]]
        # Get human next time data for collision computing
        human_gt = gt_states['states'][human_offset[0]:human_offset[1]]
        wind_gt = gt_states['wind']
        trans_gt = gt_states['trans_state']

        # Process outer force
        human_data = []
        trans_data = []
        wind_data = []
        for i in range(len(states_list['states'])-1):
            # t+1 human with t garment
            human_data.append(states_list['states'][i+1][human_offset[0]:human_offset[1]])
            trans_data.append(states_list['trans_state'][i+1])
            wind_data.append(states_list['wind'][i+1])
        # In case, non_grad_step == 0
        human_data.append(np.concatenate([human_gt, states_list['states'][-1][human_offset[0]:human_offset[1]][:, :-per_frame_states_dim['states']]], axis=-1))
        trans_data.append(np.concatenate([trans_gt, states_list['trans_state'][-1][:, :-per_frame_states_dim['trans_state']]], axis=-1))
        wind_data.append(np.concatenate([wind_gt, states_list['wind'][-1][:, :-per_frame_states_dim['wind']]], axis=-1))
        garment_data = [state[garment_offset[0]:garment_offset[-1]] for state in states_list['states']]

        names = names.astype(str)
        # Parse
        wind_data = [self._process_wind(step_wind, step_trans, trans_mean_std) for step_wind, step_trans in zip(wind_data, trans_data)]

        # p, v, a
        # easier for normalize
        standard_g = np.array([0, 0, 0, 0, 0, 0, 0, 0, -9.8]).reshape(1, -1) / self.env_cfg.layers_base.fps
        gravity = []
        for step_trans in trans_data:
            step_gravity = np.tile(standard_g, (1, self.env_cfg.history+1)) - step_trans
            step_gravity = normalize([step_gravity], [self.stat])[0]
            step_gravity = step_gravity.reshape(1, self.env_cfg.history+1, -1)
            step_gravity = step_gravity[:, :, -3:]
            step_gravity = step_gravity.reshape(1, -1)
            gravity.append(step_gravity)
        gravity = np.stack(gravity, axis=0)
        if non_grad_step <= 0:
            gravity = gravity.squeeze(0)

        garment_data = np.stack(garment_data, axis=0)
        human_data = np.stack(human_data, axis=0)
        trans_data = np.stack(trans_data, axis=0)
        wind_data = np.stack(wind_data, axis=0)
        # Get wind rotation and next wind rotation
        wind_len = wind_data.shape[0]
        wind_rots = rotation_from_quats_np(wind_data.reshape(wind_len, self.env_cfg.history+1, -1)[:, :, -5:-1])
        # Squeeze unnecessary dim
        if non_grad_step <= 0:
            garment_data = garment_data.squeeze(0)
            human_data = human_data.squeeze(0)
            trans_data = trans_data.squeeze(0)
            wind_data = wind_data.squeeze(0)
            wind_rots = wind_rots.squeeze(0)

        # Meta
        # Meta info
        indices = garment_offset
        indices_weight = np.concatenate([
            np.ones(garment_offset[i+1]-garment_offset[i]) / (garment_offset[i+1]-garment_offset[i])
            for i in range(len(garment_offset)-1)],
            axis=0)
        indices_type = g_types
        # Patchwise
        p_rows, p_cols, num_patch, patch_offset = self.patch_data(seq_num, names[:len(garment_offset)-1], garment_offset)
        g2p_mapping = defaultdict(list)
        p2g_mapping = defaultdict(list)
        for g, p in zip(p_rows, p_cols):
            g2p_mapping[g].append(p)
            p2g_mapping[p].append(g)
        g2p_mask.add(p_rows, p_cols)

        # Dynamic loading
        patch_radius = self.env_cfg.layers_base.radius
        human_radius = self.env_cfg.layers_base.human_radius
        
        # Load static
        seq_info = self.clothenv_reader.read_info(seq_num)
        mesh_name, mesh_offset, mesh_face_offset, mesh_faces = self.load_raw_static(['name', 'offset', 'face_offset', 'face'], os.path.join(self.clothenv_reader.generated_dir, seq_num, 'static.h5'), self.env_cfg.get('omit_keys', None))
        garment_faces = mesh_faces[mesh_face_offset[0]:mesh_face_offset[-2]]
        # mesh masks
        l_mask = laplacianMatrix(garment_faces)
        gmesh_mask.add(l_mask.row, l_mask.col)

        mesh_graph = edges2graph(faces2edges(garment_faces))
        for rcv_idx in mesh_graph.keys():
            sender_idx = list(mesh_graph[rcv_idx].keys())
            rcv_p_list = g2p_mapping[rcv_idx]
            sender_p_list = list(chain(*[g2p_mapping[i] for i in sender_idx]))
            pmesh_mask.add(
                np.repeat(rcv_p_list, len(sender_p_list)),
                np.tile(sender_p_list, len(rcv_p_list)))

        ## Get garemnt faces
        garment_vert2face = SparseMask('v2f')
        for f_idx, f in enumerate(garment_faces):
            garment_vert2face.add(f, np.array([f_idx]*len(f)))
        ## Get human faces
        human_faces = mesh_faces[mesh_face_offset[-2]:mesh_face_offset[-1]] - mesh_offset[-2]
        human_faces = human_faces.astype(np.int64)
        human_vert2face = SparseMask('h_v2f')
        for f_idx, f in enumerate(human_faces):
            human_vert2face.add(f, np.array([f_idx]*len(f)))

        ## Get the attributes
        attr_list = []
        garment_name = names[:len(garment_offset)-1]
        for g_idx, g_name in enumerate(garment_name):                
            attr = self.clothenv_reader.read_garment_attributes(seq_num, g_name, info=seq_info)
            # Check duplicated attr
            attr_exist = -1
            for idx, attr_alr in enumerate(attr_list):
                if np.sum(np.abs(attr - attr_alr)) <= 0:
                    attr_exist = idx
                    break
            if attr_exist < 0:
                attr_list.append(attr)
                attr_exist = len(attr_list)-1
            r_p = np.arange(garment_offset[g_idx], garment_offset[g_idx+1])
            s_p = np.ones(r_p.shape[0]) * attr_exist
            attr_mask.add(np.array(r_p), np.array(s_p))
        
        attr_list = np.stack(attr_list, axis=0)
        ## Invisable forces
        r_p = np.arange(garment_offset[0], garment_offset[-1])
        s_p =  np.zeros(r_p.shape[0])
        invisable_force_mask.add(r_p, s_p)

        # This include garment+human+attr+wind
        inputs = dict(
            # Garment
            state=DataContainer(torch.from_numpy(garment_data.astype(np.float32)).transpose(-1, -2), stack=True, padding_value=0, pad_dims=1),
            # Human
            h_state=DataContainer(torch.from_numpy(human_data.astype(np.float32)).transpose(-1, -2), stack=True, padding_value=0, pad_dims=1),
            attr=DataContainer(torch.from_numpy(attr_list.astype(np.float32)).transpose(0, 1), stack=True, padding_value=0, pad_dims=1),
            invisable_force=DataContainer(torch.from_numpy(wind_data.astype(np.float32)), stack=True, pad_dims=None),
            invisable_rotation=DataContainer(torch.from_numpy(wind_rots.astype(np.float32)), stack=True, pad_dims=None),
            # indices: n_outfit + 1
            indices=DataContainer(torch.from_numpy(indices), stack=False),
            h_indices=DataContainer(torch.from_numpy(human_offset), stack=False),
            p_indices=DataContainer(torch.from_numpy(patch_offset), stack=False),
            indices_weight=DataContainer(torch.from_numpy(indices_weight.astype(np.float32)).unsqueeze(0), stack=True, padding_value=0, pad_dims=1),
            indices_type=DataContainer(torch.from_numpy(indices_type.astype(np.float32)), stack=False),
            # For normalize
            mean_std=DataContainer(torch.from_numpy(mean_std.astype(np.float32)), stack=True, pad_dims=None),
            radius=DataContainer(torch.from_numpy(np.array(patch_radius).astype(np.float32)), stack=True, pad_dims=None),
            human_radius=DataContainer(torch.from_numpy(np.array(human_radius).astype(np.float32)), stack=True, pad_dims=None),
        )

        inputs['gravity'] = DataContainer(torch.from_numpy(gravity.astype(np.float32)), stack=True, pad_dims=None)
        # Add mask info
        # The garment is from 0 to garment_offset[-1]
        num_garment = garment_data.shape[0] if non_grad_step == 0 else garment_data.shape[1]
        g2p_mask = {
            key.replace('_sparse', ''): DataContainer(val.to_dense().transpose(-1, -2).unsqueeze(0), stack=True, padding_value=0, pad_dims=2)
            for key, val in g2p_mask.get_sparse(num_garment, num_patch, separate=False).items()
        }
        attr_mask = {
            key.replace('_sparse', ''): DataContainer(val.to_dense().transpose(-1, -2).unsqueeze(0), stack=True, padding_value=0, pad_dims=2)
            for key, val in attr_mask.get_sparse(num_garment, len(attr_list), separate=False).items()
        }
        wind_length = len(wind_data) if non_grad_step <= 0 else len(wind_data[0])
        invisable_force_mask = {
            key.replace('_sparse', ''): DataContainer(val.to_dense().transpose(-1, -2).unsqueeze(0), stack=True, padding_value=0, pad_dims=2)
            for key, val in invisable_force_mask.get_sparse(num_garment, wind_length, separate=False).items()
        }
        pmesh_mask = {
            key.replace('_sparse', ''): DataContainer(val.to_dense().transpose(-1, -2).unsqueeze(0), stack=True, padding_value=0, pad_dims=2)
            for key, val in pmesh_mask.get_sparse(num_patch, num_patch, separate=False).items()
        }
        inputs.update(g2p_mask)
        inputs.update(attr_mask)
        inputs.update(invisable_force_mask)
        inputs.update(pmesh_mask)
        # For calculate normals
        inputs['faces'] = DataContainer(torch.from_numpy(garment_faces).transpose(-1, -2), stack=True, pad_dims=1, padding_value=0)
        inputs['h_faces'] = DataContainer(torch.from_numpy(human_faces).transpose(-1, -2), stack=True, pad_dims=None)
        # Add sparse mask
        inputs.update(garment_vert2face.get_sparse(num_garment, garment_faces.shape[0], separate=False))
        human_verts_num = human_data.shape[0] if non_grad_step <= 0 else human_data.shape[1]
        inputs.update(human_vert2face.get_sparse(human_verts_num, human_faces.shape[0], separate=False))

        gt_label = dict(
            vertices=DataContainer(torch.from_numpy(garment_gt.astype(np.float32)).transpose(0, 1), stack=True, padding_value=0, pad_dims=1)
        )
        meta=dict(sequence=int(seq_num), frame=frame_idx + 1+non_grad_step, radius=patch_radius)

        input_data = dict(
            inputs=inputs,
            gt_label=gt_label,
            meta=meta,)
        return input_data

    def collate(self, batch, samples_per_gpu=1):
        # TODO: have bug when multiple inputs
        sparse_data = dict()
        for b_data in batch:
            if not isinstance(b_data['inputs'], list):
                b_data['inputs'] = [b_data['inputs']]
            for frame_idx in range(len(b_data['inputs'])):
                if frame_idx not in sparse_data.keys():
                    sparse_data[frame_idx] = defaultdict(list)
                keys = list(b_data['inputs'][frame_idx].keys())
                for key in keys:
                    if key.endswith('sparse'):
                        s_data = b_data['inputs'][frame_idx].pop(key)
                        sparse_data[frame_idx][key].append(s_data)
        for frame_idx in range(len(sparse_data)):
            for key in sparse_data[frame_idx].keys():
                batched_val = collate_sparse(sparse_data[frame_idx][key], num_head=1)
                sparse_data[frame_idx][key] = batched_val

        batched_data = collate(batch, samples_per_gpu=samples_per_gpu)

        for frame_idx in sparse_data.keys():
            batched_data['inputs'][frame_idx].update(sparse_data[frame_idx])
        if len(batched_data['inputs']) == 1:
            batched_data['inputs'] = batched_data['inputs'][0]
        return batched_data