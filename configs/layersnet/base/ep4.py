_base_ = ['./ep1.py']


non_grad_step = 3
step = 1

model = dict(
    train_cfg=dict(
        non_grad_step=non_grad_step,),)
data = dict(
    train=dict(
        env_cfg=dict(
            omit_frame=step+non_grad_step, # Since it's auto regressive
            non_grad_step=non_grad_step,),),)
runner = dict(type='EpochBasedRunner', max_epochs=4)