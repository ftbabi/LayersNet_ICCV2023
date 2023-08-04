_base_ = [
    './layers.py',
]

dataset_type = 'PatchLayersDataset'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        phase='train',
        env_cfg=dict(
            rollout=False,
            meta_path='data/data/entry_train_d.txt', # path to the "seq\tframe_num"
            max_frame=40,
            start_frame=60, # Easy way to sample frames
            omit_frame=1,
            init_frame=2, # The previous frames don't have valid vel or acc
            history=1,
            patch_size=0.04,
            step=1,
            non_grad_step=0,
            num_head=8,),),
    val=dict(
        type=dataset_type,
        phase='val',
        env_cfg=dict(
            rollout=False,
            meta_path='data/data/entry_valid_d.txt', # path to the "seq\tframe_num"
            max_frame=40,
            start_frame=60, # Easy way to sample frames
            omit_frame=1,
            init_frame=2,
            history=1,
            patch_size=0.04,
            step=1,
            non_grad_step=0,
            num_head=8,),),
    test=dict(
        type=dataset_type,
        phase='test', 
        env_cfg=dict(
            rollout=True,
            meta_path='data/data/entry_test_d.txt', # path to the "seq\tframe_num"
            max_frame=40,
            start_frame=60, # Easy way to sample frames
            omit_frame=1,
            init_frame=2,
            history=1,
            patch_size=0.04,
            step=1,
            non_grad_step=0,
            num_head=8,),),
)