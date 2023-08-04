import os
import torch
import numpy as np
import tqdm
import mmcv
from mmcv.parallel import collate, DataContainer
from collections import defaultdict

from .builder import DATASETS
from .base_dataset import BaseDataset
from layersnet.datasets.utils import diff_pos, init_stat, combine_cov_stat
from layersnet.datasets.utils import writeH5, readH5


@DATASETS.register_module()
class LayersDataset(BaseDataset):

    def __init__(self, env_cfg, phase, **kwargs):
        super(LayersDataset, self).__init__(env_cfg=env_cfg, phase=phase, **kwargs)

        # Dynamic statics
        stat_path = os.path.join(self.env_cfg.layers_base.generated_dir, 'stat.h5')
        stat = readH5(
            ['position', 'velocity', 'acceleration', 'trans_position', 'trans_velocity', 'trans_acceleration'],
            stat_path)
        cov_stat = [cs[1:4] for cs in stat]
        cov_stat = np.concatenate(cov_stat, axis=-1)
        self.cov_stat = cov_stat[:, :9]
        self.trans_cov_stat = cov_stat[:, 9:]

        stat = [self._reformat_stat(cs) for cs in stat]
        stat = np.concatenate(stat, axis=-1)
        # If the std = 0, it should be set to 1
        stat[1, np.where(stat[1, :] == 0)] = 1.0
        assert 9 == stat.shape[-1] / 2
        self.stat = stat[:, :9]
        self.trans_stat = stat[:, 9:]
        
        self.data_names = [
            'g_type',
            'name',
            'offset',
            'position', 'velocity', 'acceleration',
            'wind']
    
    def _reformat_stat(self, cov_stat):
        mean = cov_stat[0]
        std = np.sqrt(np.diag(cov_stat[1:4]))
        num = cov_stat[4]

        orig_stat = np.stack([mean, std, num], axis=0)
        return orig_stat

    def patch_data(self, sample, names_list, garment_offset):
        rows, cols = [], []
        patch_offset = [0]
        for idx, g_name in enumerate(names_list):
            patch_data = self.clothenv_reader.read_patched_garment(sample, g_name, patch_size=self.env_cfg.patch_size)
            # num_particles, num_patches
            p2p_row = patch_data['v2p_mask']['row']
            p2p_col = patch_data['v2p_mask']['col']
            rows.append(p2p_row)
            cols.append(p2p_col)
            patch_offset.append(np.max(p2p_col)+1)
        patch_offset = np.cumsum(patch_offset)
        for idx in range(len(cols)):
            cols[idx] += patch_offset[idx]
            rows[idx] += garment_offset[idx]

        rows = np.concatenate(rows, axis=0)
        cols = np.concatenate(cols, axis=0)

        num_patch = np.max(cols) + 1
        assert num_patch == patch_offset[-1]

        return rows, cols, num_patch, patch_offset

    def generate_data(self, save_dir=None, seq=None):
        data_dir = os.path.join(self.clothenv_reader.data_dir, 'data')
        if save_dir is None:
            save_dir = self.clothenv_reader.generated_dir
        sample_list = os.listdir(data_dir)

        # Only calculate mean_std in training set
        training_seq = set()
        if self.phase == 'train':
            training_seq = set([i[0] for i in self.data_list])

        # Pos, vel, acc
        # mean 1*3, std 3*3, n 1*3 
        stats = [init_stat(dim=3, entry=5), init_stat(dim=3, entry=5), init_stat(dim=3, entry=5)]
        # Pos, vel, acc for human trans; only for relative input usages
        trans_stats = [init_stat(dim=3, entry=5), init_stat(dim=3, entry=5), init_stat(dim=3, entry=5)]

        if seq is not None:
            sample_list = [seq]
        for sample_idx, sample_num in enumerate(sample_list):
            if not os.path.isdir(os.path.join(data_dir, sample_num)):
                continue
            print(f"Processing {sample_num} {sample_idx}/{len(sample_list)}")
            sample_info = self.clothenv_reader.read_info(sample_num)
            dt = self.env_cfg.layers_base.dt
            outer_forces = sample_info['wind']
            assert len(outer_forces) == 1

            num_frame = sample_info['human']['seq_end'] - sample_info['human']['seq_start']

            outer_forces_idx = [
                o['frame_start'] - sample_info['human']['frame_start']
                for o in outer_forces[0]['pivot_list']] \
                    + [num_frame]

            human_pos = [self.clothenv_reader.read_human(sample_num, i)[0] for i in range(sample_info['human']['seq_start'], sample_info['human']['seq_end'])]
            human_pos = np.stack(human_pos, axis=0)
            # Deal with global human movement
            ## num_frame, 3
            global_trans = np.stack([self.clothenv_reader.read_smpl_params(sample_num, frame)[-1].reshape(1, -1) for frame in range(num_frame)], axis=0)
            garment_pos_list = []
            garment_names = []
            garment_types = []
            for g_meta in sample_info['garment']:
                g_name = g_meta['name']
                garment_names.append(g_name)
                garment_types.append(g_meta['type'])
                garment_pos = self.clothenv_reader.read_garment_vertices(sample_num, g_name)
                if garment_pos is None:
                    # Invalid sim data
                    continue
                garment_pos_list.append(garment_pos)
            
            # Prepare init velocities and acceleories
            if len(human_pos) < 2:
                continue
            human_vel, human_acc = diff_pos(human_pos, dt=dt, padding=True)
            global_vel, global_acc = diff_pos(global_trans, dt=dt, padding=True)
            garment_vel_list, garment_acc_list = [], []
            for g_pos in garment_pos_list:
                g_vel, g_acc = diff_pos(g_pos, dt=dt, padding=True)
                garment_vel_list.append(g_vel)
                garment_acc_list.append(g_acc)
            
            seq_length = human_acc.shape[0]
            for frame_idx in tqdm.tqdm(range(seq_length)):
                pos_list = []
                vel_list = []
                acc_list = []
                order_list = []
                index_list = [0]
                # Garment
                for g_name, g_pos, g_vel, g_acc in zip(garment_names, garment_pos_list, garment_vel_list, garment_acc_list):
                    pos_list.append(g_pos[frame_idx])
                    vel_list.append(g_vel[frame_idx])
                    acc_list.append(g_acc[frame_idx])
                    order_list.append(g_name)
                    index_list.append(g_pos[frame_idx].shape[0])
                # Human
                pos_list.append(human_pos[frame_idx])
                vel_list.append(human_vel[frame_idx])
                acc_list.append(human_acc[frame_idx])
                order_list.append('human')
                index_list.append(human_pos[frame_idx].shape[0])
                # Global human pos
                pos_list.append(global_trans[frame_idx])
                vel_list.append(global_vel[frame_idx])
                acc_list.append(global_acc[frame_idx])
                order_list.append('global')
                index_list.append(global_trans[frame_idx].shape[0]) # in case it's 1
                # Outer forces
                ## Padding for zeros
                if frame_idx < outer_forces_idx[0]:
                    w_info = np.zeros(5)
                else:
                    for o_idx in range(1, len(outer_forces_idx)):
                        assert outer_forces_idx[o_idx-1] <= outer_forces_idx[o_idx]
                        if frame_idx < outer_forces_idx[o_idx] and frame_idx >= outer_forces_idx[o_idx-1]:
                            # find it
                            w_meta = outer_forces[0]['pivot_list'][o_idx-1]
                            rotations = w_meta['rotation_quaternion']
                            strengths = np.array(w_meta['strength']).reshape(1)
                            w_info = np.concatenate([rotations, strengths], axis=0)
                            break

                offset_idx = np.cumsum(index_list)
                
                # Relative positions and vel and acc
                pos_list = np.concatenate(pos_list, axis=0)
                vel_list = np.concatenate(vel_list, axis=0)
                acc_list = np.concatenate(acc_list, axis=0)
                pos_list[:offset_idx[-2]] -= pos_list[offset_idx[-2]:]
                vel_list[:offset_idx[-2]] -= vel_list[offset_idx[-2]:]
                acc_list[:offset_idx[-2]] -= acc_list[offset_idx[-2]:]
                ## Normalize
                if sample_num in training_seq:
                    candicates = [pos_list[:offset_idx[-2]], vel_list[:offset_idx[-2]], acc_list[:offset_idx[-2]]]
                    for j in range(len(stats)):
                        stat = init_stat(dim=stats[j].shape[1], entry=stats[j].shape[0])
                        stat[0, :] = np.mean(candicates[j], axis=0)
                        stat[1:4, :] = self.covariance(candicates[j], stat[0, :])
                        stat[4, :] = candicates[j].shape[0]
                        stats[j] = combine_cov_stat(stats[j], stat)
                    # For global trans
                    g_candidates = [pos_list[offset_idx[-2]:], vel_list[offset_idx[-2]:], acc_list[offset_idx[-2]:]]
                    for j in range(len(trans_stats)):
                        stat = init_stat(dim=trans_stats[j].shape[1], entry=trans_stats[j].shape[0])
                        stat[0, :] = np.mean(g_candidates[j], axis=0)
                        stat[1:4, :] = self.covariance(g_candidates[j], stat[0, :])
                        stat[4, :] = g_candidates[j].shape[0]
                        trans_stats[j] = combine_cov_stat(trans_stats[j], stat)

                rollout_dir = os.path.join(save_dir, sample_num, "rollout")
                mmcv.mkdir_or_exist(rollout_dir)
                save_path = os.path.join(rollout_dir, f"{frame_idx}.h5")
                # Encoding
                order_list = [i.encode('utf8') for i in order_list]
                data = [
                    garment_types,
                    order_list,
                    offset_idx,
                    pos_list, vel_list, acc_list,
                    w_info]
                writeH5([
                    'g_type',
                    'name',
                    'offset',
                    'position', 'velocity', 'acceleration',
                    'wind'], data, save_path)

        if seq is not None:
            return
        # Save global stat
        stat_path = os.path.join(save_dir, "stat.h5")
        print(f"Saving global stats to {stat_path}")
        writeH5(
            ['position', 'velocity', 'acceleration',
            'trans_position', 'trans_velocity', 'trans_acceleration'],
            stats + trans_stats, stat_path)
        return
    
    def generate_static_data(self, save_dir=None, seq=None):
        data_dir = os.path.join(self.clothenv_reader.data_dir, 'data')
        if save_dir is None:
            save_dir = self.clothenv_reader.generated_dir
        sample_list = os.listdir(data_dir)

        if seq is not None:
            sample_list = [seq]
        for sample_idx, sample_num in enumerate(sample_list):
            if not os.path.isdir(os.path.join(data_dir, sample_num)):
                continue
            print(f"Processing {sample_num} {sample_idx}/{len(sample_list)}")
            sample_info = self.clothenv_reader.read_info(sample_num)
            garment_names = []
            garment_offset = [0]
            garment_face_offset = [0]
            garment_faces = []
            for g_meta in sample_info['garment']:
                g_name = g_meta['name']
                garment_names.append(g_name)
                
                garment_pos = self.clothenv_reader.read_garment_vertices(sample_num, g_name, frame=0)
                if garment_pos is None:
                    # Invalid sim data
                    continue
                garment_offset.append(garment_pos.shape[0])

                F, _ = self.clothenv_reader.read_garment_topology(sample_num, g_name)
                garment_faces.append(F)
                garment_face_offset.append(F.shape[0])
            # Add human faces
            human_verts, human_faces = self.clothenv_reader.read_human(sample_num, frame=0)
            garment_names.append("human")
            garment_offset.append(human_verts.shape[0])
            garment_faces.append(human_faces)
            garment_face_offset.append(human_faces.shape[0])

            garment_offset = np.cumsum(garment_offset)
            for i in range(len(garment_offset)-1):
                garment_faces[i] += garment_offset[i]
            garment_faces = np.concatenate(garment_faces, axis=0)
            garment_face_offset = np.cumsum(garment_face_offset)

            save_seq_dir = os.path.join(save_dir, sample_num)
            mmcv.mkdir_or_exist(save_seq_dir)
            save_path = os.path.join(save_seq_dir, "static.h5")
            print(f"Saving static info to {save_path}")
            writeH5(
                ['name', 'offset', 'face_offset', 'face'],
                [garment_names, garment_offset, garment_face_offset, garment_faces],
                save_path
            )
    
    def covariance(self, data, mean):
        # return std^2
        n_verts, n_dim = data.shape
        # mean.shape = 1, 3
        cov_data = data - mean.reshape(1, n_dim)
        cov_data = cov_data.reshape(n_verts, n_dim, 1)
        # n_verts, n_dim, n_dim
        cov_data = np.matmul(cov_data, cov_data.transpose(0, 2, 1))
        cov_data = np.sum(cov_data, axis=0) / n_verts
        return cov_data