from sys import prefix
import numpy as np
import torch


class SparseMask:
    def __init__(self, prefix=''):
        if prefix:
            prefix = f"{prefix}_"
        self.prefix = prefix
        self.suffix = "_sparse"

        self.receiver_idx = []
        self.sender_idx = []
        self.relation_idx = []

    def add(self, r_idx, s_idx, e_idx=None):
        if isinstance(r_idx, list):
            r_idx = np.array(r_idx)
        if isinstance(s_idx, list):
            s_idx = np.array(s_idx)

        assert r_idx.shape[0] == s_idx.shape[0]

        self.receiver_idx.append(r_idx)
        self.sender_idx.append(s_idx)
        if e_idx is not None:
            if isinstance(e_idx, list):
                e_idx = np.array(e_idx)
            self.relation_idx.append(e_idx)
    
    def get_sparse(self, row_size, col_size, separate=True, dtype=np.float32):
        r_idx = np.concatenate(self.receiver_idx, axis=0)
        s_idx = np.concatenate(self.sender_idx, axis=0)
        if separate:
            if len(self.relation_idx) <= 0:
                self.relation_idx.append(np.arange(r_idx.shape[0]))
            e_idx = np.concatenate(self.relation_idx, axis=0)
            rst = {
                f"{self.prefix}r_mask{self.suffix}": torch.sparse_coo_tensor(
                    np.stack([r_idx, e_idx], axis=0),
                    np.ones(r_idx.shape[0]).astype(dtype),
                    size=(row_size, e_idx.shape[0])).coalesce(),
                f"{self.prefix}s_mask{self.suffix}": torch.sparse_coo_tensor(
                    np.stack([s_idx, e_idx], axis=0),
                    np.ones(s_idx.shape[0]).astype(dtype),
                    size=(col_size, e_idx.shape[0])).coalesce(),
            }
        else:
            rst = {
                f"{self.prefix}mask{self.suffix}": torch.sparse_coo_tensor(
                    np.stack([r_idx, s_idx], axis=0),
                    np.ones(r_idx.shape[0]).astype(dtype),
                    size=(row_size,col_size)).coalesce(),
            }
        
        return rst
        

def collate_sparse(batch_list, samples_per_gpu=1, num_head=1):
    # 2 * n
    indices = [i.indices() for i in batch_list]
    data = [i.values().tile((num_head,)) for i in batch_list]
    sizes = np.array([i.size() for i in batch_list])
    max_sizes = np.max(sizes, axis=0)

    # Collate indices
    collated_indices = []
    # This is collate for collapse of batchsize dim and head dim.
    ## bs*num_head, n, dim
    for idx, ind in enumerate(indices):
        for h_idx in range(num_head):
            new_axis = torch.full((1, ind.shape[1]), idx*num_head + h_idx)
            cur_ind = torch.concat([new_axis, ind], axis=0)
            collated_indices.append(cur_ind)
    s_size = (len(batch_list)*num_head, *max_sizes.tolist())
    # 3 * n_total
    collated_indices = torch.concat(collated_indices, axis=1)
    collated_data = torch.concat(data, axis=0)

    collated_sparse_data = dict(
        indices=collated_indices,
        values=collated_data,
        size=s_size,
    )
    return collated_sparse_data