_base_ = [
    '../../_base_/models/layersnet.py',
    '../../_base_/datasets/patch_layers.py',
    '../../_base_/schedules/adam_step_bs256.py',
    '../../_base_/default_runtime.py'
]

find_unused_parameters = True

non_grad_step = 0
model = dict(
    train_cfg=dict(
        non_grad_step=non_grad_step,),
    )

# Custom dataset
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=16,)

# Custom scheduler
runner = dict(type='EpochBasedRunner', max_epochs=1)