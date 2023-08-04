from abc import ABCMeta
import os
from itertools import chain
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset

from .utils import LayersReader, GARMENT_TYPE, COLLATE_GARMENT_TYPE


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.
    """

    def __init__(self, env_cfg, phase, **kwargs):
        self.env_cfg = env_cfg
        self.phase = phase
        # Usefull when testing
        self.rollout = env_cfg.get('rollout', False)
        self._init_reader()
        # Mapping for each frame in each sequence
        self.data_list, self.rollout_list = self.decode_meta(
            self.env_cfg.meta_path,
            omit_frame=self.env_cfg.omit_frame,
            init_framne=self.env_cfg.init_frame,
            history=self.env_cfg.history)
        self.supervised_rollout = env_cfg.get("supervised_rollout", False)
        self.non_grad_step = env_cfg.get("non_grad_step", 0)
    
    def set_sequence(self, length):
        self.non_grad_step = length
    
    def _init_reader(self):
        self.clothenv_reader = LayersReader(self.env_cfg.layers_base, phase=self.phase)

    def filter_data(self, seq_list=[]):
        data_list = []
        for seq in seq_list:
            for i in self.data_list:
                if int(i[0]) == int(seq):
                    data_list.append(i)
        self.data_list = data_list

    def prepare_rollout(self, batch_data, **kwargs):
        return batch_data

    def decode_meta(self, sheet_path, omit_frame=0, init_framne=0, history=0):
        if not os.path.exists(sheet_path):
            self.preprocess(sheet_path)

        with open(sheet_path, 'r') as f:
            sheet_list = f.readlines()
        data_list = []
        rollout_list = []
        for line in sheet_list:
            seq, seq_start, seq_end = line.split('\t')
            seq_start = int(seq_start)
            seq_end = int(seq_end)
            if self.env_cfg.get('start_frame', None) is not None:
                seq_start = max(seq_start, self.env_cfg.start_frame)
            seq_len = seq_end - seq_start
            if self.env_cfg.get('max_frame', None) is not None:
                seq_len = min(seq_len, self.env_cfg.max_frame)
            valid_num_frames = seq_len-omit_frame-init_framne-history
            if valid_num_frames <= 0:
                continue
            seq_list = [(seq, i+init_framne+history+seq_start) for i in range(valid_num_frames)]
            data_list.extend(seq_list)
            rollout_list.append((seq, valid_num_frames))
        
        return data_list, rollout_list
    
    def preprocess(self, sheet_path):
        # sheet format: seq\tnum
        sample_seq = self.clothenv_reader.seq_list

        data_list = []
        for seq in sample_seq:
            sample_info = self.clothenv_reader.read_info(seq)
            num_frames = sample_info['human']['seq_end'] - sample_info['human']['seq_start']
            data_list.append(f"{seq}\t0\t{num_frames}")
        with open(sheet_path, 'w') as f:
            f.write("\n".join(data_list))
        return True

    def __len__(self):
        length = len(self.data_list)
        return length

    def evaluate_frame(self,
                 results,
                 metric=None,
                 logger=None,
                 **kwargs):
        eval_results = dict()
        for rst in results:
            for g_name, g_rst in rst.items():
                if g_name not in eval_results.keys():
                    eval_results[g_name] = []
                if g_rst > 0:
                    eval_results[g_name].append(g_rst)
        
        for g_name in eval_results.keys():
            if len(eval_results[g_name]) < 1:
                eval_results[g_name].append(0)
        
                # Mean results
        collate_results = dict()
        for g_name, g_list in eval_results.items():
            collate_results[g_name] = dict(
                mean=np.mean(g_list),
                std=np.std(g_list),
            )

        return collate_results

    def evaluate_rollout_each_metric(self,
                 results,
                 metric_key,
                 extra_key=None,
                 ignore_key=[],
                 merged=False,
                 logger=None,
                 **kwargs):
        collate_results = dict()
        if isinstance(ignore_key, str):
            ignore_key = [ignore_key]

        garment_wise_results = dict()
        rollout_wise_results = dict()
        # Collect results
        for rst in results:
            rollout_idx = rst['rollout_idx']
            if not rollout_idx in rollout_wise_results.keys():
                rollout_wise_results[rollout_idx] = []
            indices_name = []
            for g_idx, i_type in enumerate(rst['indices_type']):
                g_type_idx = np.argmax(i_type)
                g_type = GARMENT_TYPE[g_type_idx]
                indices_name.append(g_type)
            if extra_key is not None:
                indices_name.append(extra_key) # This is for collision default keys
            rollout_wise_results[rollout_idx].append(
                dict(acc=rst['acc'], indices=rst['indices'], indices_name=indices_name))
            for g_name, g_rst in rst['acc'].items():
                ignore_this = False
                for ig_key in ignore_key:
                    if ig_key in g_name:
                        ignore_this = True
                        break
                if ignore_this:
                    continue
                if not g_name.replace(f'{metric_key}.', '') in indices_name:
                    continue
                if g_rst < 0:
                    continue
                if g_name not in garment_wise_results.keys():
                    garment_wise_results[g_name] = defaultdict(list)
                garment_wise_results[g_name][rollout_idx].append(g_rst)

        # Calculate rollout level results
        eval_rollout = dict()
        extra_eval_rollout = dict()
        for rollout_idx, rollout_cont in rollout_wise_results.items():
            rollout_rst = []
            extra_rst = []
            for frame_cont in rollout_cont:
                indices = frame_cont['indices']
                indices_name = frame_cont['indices_name']
                frame_eval = 0
                extra_key_eval = 0
                for g_idx, g_type in enumerate(indices_name):
                    if g_type == extra_key:
                        num = indices[-1] - indices[0]
                        extra_key_eval += frame_cont['acc'][f'{metric_key}.{g_type}'] * num
                    else:
                        num = indices[g_idx+1] - indices[g_idx]
                        if frame_cont['acc'][f'{metric_key}.{g_type}'] < 0:
                            # To fit for collision
                            continue
                        frame_eval += frame_cont['acc'][f'{metric_key}.{g_type}'] * num
                frame_eval /= indices[-1]
                extra_key_eval /= indices[-1]
                rollout_rst.append(frame_eval)
                extra_rst.append(extra_key_eval)
            eval_rollout[rollout_idx] = rollout_rst
            extra_eval_rollout[rollout_idx] = extra_rst
        collate_results[f'{metric_key}_whole'] = dict(
            mean=np.mean(list(chain(*[i for i in eval_rollout.values()]))),
            std=np.std(list(chain(*[i for i in eval_rollout.values()])))
        )
        collate_results[f'{metric_key}_rollout'] = dict(
            mean=np.mean([np.mean(i) for i in eval_rollout.values()]),
            std=np.std([np.mean(i) for i in eval_rollout.values()])
        )

        # Save into logger
        for rollout_idx, rollout_rst in eval_rollout.items():
            logger.info(f"Rollout {rollout_idx} {metric_key}: mean: {np.mean(rollout_rst)}; std: {np.std(rollout_rst)}")

        if extra_key is not None:
            collate_results[f'{metric_key}_whole_{extra_key}'] = dict(
                mean=np.mean(list(chain(*[i for i in extra_eval_rollout.values()]))),
                std=np.std(list(chain(*[i for i in extra_eval_rollout.values()])))
            )
            collate_results[f'{metric_key}_rollout_{extra_key}'] = dict(
                mean=np.mean([np.mean(i) for i in extra_eval_rollout.values()]),
                std=np.std([np.mean(i) for i in extra_eval_rollout.values()])
            )
            # Save into logger
            for rollout_idx, rollout_rst in extra_eval_rollout.items():
                logger.info(f"Rollout {rollout_idx} {metric_key}: mean: {np.mean(rollout_rst)}; std: {np.std(rollout_rst)}")

        # Garment wise results
        if merged:
            merged_results = dict()
            garment_keys = list(garment_wise_results.keys())
            for key in garment_keys:
                collated_key = None
                for candi in COLLATE_GARMENT_TYPE:
                    if candi in key:
                        collated_key = f'{metric_key}_{candi}'
                        break
                if collated_key is None:
                    collated_key = key
                    logger.info(f"Warning: {key} is a new one")
                if collated_key not in merged_results.keys():
                    merged_results[collated_key] = dict()
                for rollout_idx, cont in garment_wise_results[key].items():
                    assert rollout_idx not in merged_results[collated_key]
                    merged_results[collated_key][rollout_idx] = cont
            garment_wise_results = merged_results
            
        for g_name, rollout_cont in garment_wise_results.items():
            rollout_eval = {
                rollout_idx: np.mean(rollout_acc)
                for rollout_idx, rollout_acc in rollout_cont.items()
            }
            collate_results[g_name] = dict(
                whole_mean=np.mean(list(chain(*rollout_cont.values()))), # frame wise results
                whole_std=np.std(list(chain(*rollout_cont.values()))), # frame wise results
                rollout_mean=np.mean(list(rollout_eval.values())),
                rollout_std=np.std(list(rollout_eval.values())),
            )

        return collate_results
    
    def evaluate_rollout(self,
                 results,
                 metric=None,
                 metric_options=None,
                 logger=None,
                 **kwargs):
        collate_results = dict()

        if metric_options is None:
            metric_options = dict()
        collate_results.update(
            self.evaluate_rollout_each_metric(results=results, metric_key='accuracy_l2', 
            merged=self.env_cfg.get('merged', False), logger=logger, **metric_options, **kwargs))
        if self.env_cfg.get('collision', False):
            collate_results.update(
                self.evaluate_rollout_each_metric(results=results, metric_key='accuracy_collision', extra_key='garment2human', merged=self.env_cfg.get('merged', False), logger=logger, **metric_options, **kwargs))

        return collate_results

    def evaluate(self,
                 results,
                 metric=None,
                 metric_options=None,
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.
        """
        if self.env_cfg.get("rollout", False):
            return self.evaluate_rollout(results=results, metric=metric, metric_options=metric_options, logger=logger, **kwargs)
        else:
            return self.evaluate_frame(results=results, metric=metric, metric_options=metric_options, logger=logger, **kwargs)