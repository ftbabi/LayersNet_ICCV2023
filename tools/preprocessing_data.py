import argparse
import os.path as osp
import time

import mmcv
from mmcv import DictAction
from layersnet.datasets import build_dataset
from layersnet.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='LayersNet')
    parser.add_argument('config', help='The config file path')
    parser.add_argument('--work_dir', type=str, default='', help='The log folder')
    parser.add_argument('--dataset', type=str, default='train', help='Dataset type. One of [train/val/test]')
    parser.add_argument('--type', type=str, default='dynamic', help='dynamic or static')
    parser.add_argument('--save_dir', type=str, default=None, help='Save data folder')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # create work_dir
    work_dir = args.work_dir
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    logger.info(f'Config:\n{cfg.pretty_text}')
    
    logger.info('Save path {}'.format(work_dir))

    # build the dataloader
    if args.dataset == 'train':
        # Train data
        logger.info('Generate training data')
        cfg.data.train['is_init'] = True
        dataset = build_dataset(cfg.data.train)
    elif args.dataset == 'val':
        # Val data
        logger.info('Generate validation data')
        cfg.data.val['is_init'] = True
        dataset = build_dataset(cfg.data.val)
    elif args.dataset == 'test':
        # Val data
        logger.info('Generate test data')
        cfg.data.test['is_init'] = True
        dataset = build_dataset(cfg.data.test)
    
    if args.type == 'dynamic':
        dataset.generate_data(save_dir=args.save_dir)
    else:
        dataset.generate_static_data(save_dir=args.save_dir)

    logger.info('Finish')


if __name__ == '__main__':
    main()
