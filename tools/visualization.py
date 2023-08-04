import os
import argparse
from mmcv import Config

import numpy as np
from layersnet.utils import MeshViewer
from layersnet.datasets.utils import LayersReader, readPKL


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize output')
    parser.add_argument('config', help='the config')
    parser.add_argument('--work_dir', help='the dir to the output rollout dir')
    parser.add_argument('--seq', type=int, help='the target sequence')
    parser.add_argument('--frame', type=int, help='the target frame')

    args = parser.parse_args()

    return args


def visualize_gt(layers_reader, sample, frame, mesh_viewer=None):
    if mesh_viewer is None:
        mesh_viewer = MeshViewer()
    g_info = layers_reader.read_info(sample)

    human_V, human_F = layers_reader.read_human(sample, frame=frame)
    mesh_viewer.add_mesh(human_V, human_F) 
    for g_idx, g_cfg in enumerate(g_info['garment']):
        g_V, g_F, _ = layers_reader.read_garment_vertices_topology(sample, g_cfg['name'], frame)
        mesh_viewer.add_mesh(g_V, g_F)
    mesh_viewer.show()
    return


def visualize_pred(layers_reader, sample, frame, g_offset, pred_dir, mesh_viewer=None):
    if mesh_viewer is None:
        mesh_viewer = MeshViewer()
    g_offset = np.cumsum(g_offset)
    g_info = layers_reader.read_info(sample)

    human_V, human_F = layers_reader.read_human(sample, frame=frame)
    mesh_viewer.add_mesh(human_V, human_F)
    _, _, _, trans = layers_reader.read_smpl_params(sample, frame=frame)

    pred_path = os.path.join(pred_dir, sample, f"{frame}".zfill(3) + ".pkl")
    predictions = readPKL(pred_path)
    predictions = predictions['vertices']
    for g_idx, g_cfg in enumerate(g_info['garment']):
        g_F, _ = layers_reader.read_garment_topology(sample, g_cfg['name'])
        g_V = predictions[g_offset[g_idx]:g_offset[g_idx+1]] + trans
        mesh_viewer.add_mesh(g_V, g_F)
    mesh_viewer.show()
    return


def prepare_garment_info(layers_reader, sample):
    g_offset = [0]
    g_info = layers_reader.read_info(sample)
    for g_idx, g_cfg in enumerate(g_info['garment']):
        # Since this is static, thus use the first frame
        g_V, g_F, _ = layers_reader.read_garment_vertices_topology(sample, g_cfg['name'], frame=0)
        g_offset.append(g_V.shape[0])
    return g_offset

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # Get dataset handler
    layers_reader = LayersReader(cfg.data.test.env_cfg.layers_base)
    m_viewer = MeshViewer()
    # Give seq, num_frame
    sample = f"{args.seq}".zfill(5)
    frame = args.frame

    # Visualize gt
    # visualize_gt(
    #     layers_reader, sample, frame, mesh_viewer=m_viewer)
    
    # Visualize prediction
    ## Prepare garment info
    g_offset = prepare_garment_info(layers_reader, sample)
    pred_dir = args.work_dir
    visualize_pred(
        layers_reader, sample, frame,
        g_offset, pred_dir, mesh_viewer=m_viewer)


if __name__ == '__main__':
    main()