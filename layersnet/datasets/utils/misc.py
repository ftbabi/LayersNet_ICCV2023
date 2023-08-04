import torch
import numpy as np


def rand_int(lo, hi):
    return np.random.randint(lo, hi)

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def combine_stat(stat_0, stat_1):
    '''
        inputs are row vector, function is to process col vector
        output also need row vector
        std^2: 1/N, where N is the number of particles.
    '''
    stat_0 = stat_0.transpose()
    stat_1 = stat_1.transpose()
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    rst = np.stack([mean, std, n], axis=-1)
    return rst.transpose()

def combine_cov_stat(stat_0, stat_1):
    '''
        inputs are row vector, function is to process col vector
        output also need row vector
        std^2: 1/N, where N is the number of particles.
        stat: 5*3, 1*3 mean, 3*3 std, 1*3 n

        The return is not std. It's var. Since cov may < 0
    '''
    stat_0 = stat_0.transpose()
    stat_1 = stat_1.transpose()
    mean_0, var_0, n_0 = stat_0[:, 0], stat_0[:, 1:4], stat_0[:, 4]
    mean_1, var_1, n_1 = stat_1[:, 0], stat_1[:, 1:4], stat_1[:, 4]

    # New mean for each dim
    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    # 3*1, 1*3 -> 3*3
    cm_0 = (mean_0 - mean).reshape(3, 1)
    cm_1 = (mean_1 - mean).reshape(3, 1)
    cov_mean_0 = np.dot(cm_0, cm_0.T)
    cov_mean_1 = np.dot(cm_1, cm_1.T)
    var = (var_0 * n_0 + var_1 * n_1 + \
                   cov_mean_0 * n_0 + cov_mean_1 * n_1) / (n_0 + n_1)
    n = n_0 + n_1

    rst = np.concatenate([mean.reshape(3, 1), var, n.reshape(3, 1)], axis=-1)
    return rst.transpose()


def init_stat(dim, entry=3):
    # mean, std, count
    return np.zeros((dim, entry)).transpose()


def normalize(data, stat):
    '''
        input stat are row vector, function is to process col vector
        output also need row vector
    '''
    stat = [i.transpose(-1, -2) for i in stat]
    for i in range(len(stat)):
        if isinstance(data[i], torch.Tensor):
            # Preprocessed in dataset
            # stat[i][stat[i][:, :, 1] == 0, 1] = 1.
            stat_dim = stat[i].shape[1]
            n_rep = int(data[i].shape[-1] / stat_dim)
            bs, n_particle, data_dim = data[i].shape
            data[i] = data[i].reshape(bs, n_particle, n_rep, stat_dim)

            data[i] = (data[i] - stat[i][:, :, 0].unsqueeze(1).unsqueeze(1)) / stat[i][:, :, 1].unsqueeze(1).unsqueeze(1)
            data[i] = data[i].reshape(bs, n_particle, data_dim)
        elif isinstance(data[i], np.ndarray):
            stat[i][stat[i][:, 1] == 0, 1] = 1.
            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].shape[-1] / stat_dim)
            n_particle, data_dim = data[i].shape
            data[i] = data[i].reshape(n_particle, n_rep, stat_dim)

            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]
            data[i] = data[i].reshape(n_particle, data_dim)

    return data


def denormalize(data, stat, mask=None, mean=True):
    '''
        input stat are row vector, function is to process col vector
        output also need row vector
    '''
    stat = [i.transpose(-1, -2) for i in stat]
    for i in range(len(stat)):
        if isinstance(data[i], torch.Tensor):
            # data[i]: bs, n_particles, 3
            # stat[i]: bs, 3, 2
            data[i] = data[i] * (stat[i][:, :, 1].unsqueeze(1))
            if mean:
                data[i] += stat[i][:, :, 0].unsqueeze(1)
            # mask: bs, n_particles
            if mask is not None:
                assert isinstance(mask, torch.Tensor)
                data[i] = data[i] * mask.unsqueeze(2)
        elif isinstance(data[i], np.ndarray):
            data[i] = data[i] * stat[i][:, 1]
            if mean:
                data[i] += stat[i][:, 0]
            assert isinstance(mask, np.ndarray)
            assert data[i].shape[0] == mask.shape[0]
            if mask is not None:
                data[i] = data[i] * mask

    return data

def to_tensor_cuda(data):
    cuda_data = dict()
    for key, val in data.items():
        if isinstance(val, list):
            cuda_list = [i.unsqueeze(0).cuda() for i in val]
            cuda_data[key] = cuda_list
        else:
            cuda_data[key] = val.unsqueeze(0).cuda()
    
    return cuda_data

def to_numpy_detach(data):
    if isinstance(data, dict):
        return {
            key: to_numpy_detach(val)
            for key, val in data.items()
        }
    elif isinstance(data, list):
        return [to_numpy_detach(i) for i in data]
    else:
        data = data.detach().cpu().numpy()
        if len(data.shape) > 0:
            data = list(data)
        return data

def diff_pos(data, dt=1.0/30, padding=True):
    # Pos: [0, num_frame)
    # vel: [1, num_frame)
    # acc: [2, num_frame)
    num_frame, n_particles, dim = data.shape
    velocities = (data[1:] - data[:-1]) / dt
    if padding:
        velocities = np.concatenate([np.zeros((1, n_particles, dim)), velocities], axis=0)

    acc = (velocities[1:] - velocities[:-1]) / dt
    if padding:
        acc = np.concatenate([np.zeros((1, n_particles, dim)), acc], axis=0)

    return velocities, acc
