import argparse
import os
from textwrap import indent
import warnings
import time

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from layersnet.apis import multi_gpu_test, single_gpu_test
from layersnet.datasets import build_dataloader, build_dataset
from layersnet.models import build_simulator
from layersnet.utils import (auto_select_device, get_root_logger,
                         setup_multi_processes, wrap_distributed_model,
                         wrap_non_distributed_model)


def parse_args():
    parser = argparse.ArgumentParser(description='layersnet test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`. rollout=INDEX')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # assert args.metrics or args.out, \
    #     'Please specify at least one of output path and evaluation metrics.'

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # cfg.data.test.test_mode = True

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = args.device or auto_select_device()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if cfg.data.samples_per_gpu > 1:
            Warning(f"cfg.data.samples_per_gpu must be 1")
            cfg.data.samples_per_gpu = 1
    else:
        distributed = True
        if not cfg.data.test.env_cfg.rollout:
            raise NotImplementedError(f"Distributed test on rollout is not supported")
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # Customized: select target rollout <<<
    show_kwargs = args.show_options or {}
    specific_rollout = show_kwargs.pop('rollout', None)
    if specific_rollout is not None:
        if not isinstance(specific_rollout, list):
            specific_rollout = [int(specific_rollout)]
        else:
            specific_rollout = [int(i) for i in specific_rollout]
        dataset.filter_data(specific_rollout)
    # >>>
    
    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=False,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_simulator(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = wrap_non_distributed_model(
            model, device=cfg.device, device_ids=cfg.gpu_ids)
        show_kwargs = args.show_options or {}
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = wrap_distributed_model(
            model,
            device=cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        # work_dir is determined in this priority: CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = os.path.join('./work_dirs',
                                    os.path.splitext(os.path.basename(args.config))[0])
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file)
        # if args.metrics:
        eval_results = dataset.evaluate(
            results=outputs,
            metric=args.metrics,
            metric_options=args.metric_options,
            logger=logger)
        for key, val in eval_results.items():
            print(f'\n{key} accuracy: {val}')
        # else:
        #     raise NotImplementedError
            
    if args.out and rank == 0:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(eval_results, args.out, indent=4)


if __name__ == '__main__':
    main()
