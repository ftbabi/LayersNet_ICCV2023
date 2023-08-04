## optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=None)

## learning policy
lr_config = dict(policy='step', step=[2, 4, 6, 8, 10], gamma=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=10)