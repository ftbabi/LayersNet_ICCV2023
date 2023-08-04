model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='LayersNet',
        attr_dim=5,
        state_dim=6*(1+1), # History is one
        normal_dim=3,
        invisible_forces=(5+6)*(1+1), # cat the global states: vel and acc
        gravity_dim=3*(1+1), # only the acc
        position_dim=3,
        embed_dims=128, 
        num_heads=8,
        num_fcs=2,
        num_encoder_layers=4,
        dropout=0.0,
        order=('selfattn', 'norm', 'ffn', 'norm'),
        act_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='LN'),
        patch_neighbor=4,
        group_cfg=dict(
            max_radius=None,
            min_radius=0.0,
            sample_num=1,
            use_xyz=True,
            normalize_xyz=False,
            return_grouped_xyz=False,
            return_grouped_idx=True,
            return_unique_cnt=False,),
        gh_group_cfg=dict(
            # should be the same as base/dataset/layers.py
            max_radius=0.6,
            min_radius=0.0,
            sample_num=16, # maximum number
            use_xyz=True,
            normalize_xyz=False,
            return_grouped_xyz=False,
            return_grouped_idx=True,
            return_unique_cnt=False,),),
    decode_head=dict(
        type='PatchDecoder',
        in_channels=6*(1+1)+128*2, # particle, patch, human
        out_channels=3,
        add_residual=False,
        dt=1.0,
        loss_decode=[
            dict(type='L2Loss', reduction='sum', loss_weight=1.0, obj_wise=False),
            dict(type='CollisionLoss', reduction='sum', loss_weight=1.3, sample_num=1, radius=None, per_garment=True, per_garment_weight=0.1),
            dict(type='NormalLoss', reduction='sum', loss_weight=1.0, obj_wise=False),
            ],
        accuracy=dict(type='L2Accuracy', reduction='mean'),),
    train_cfg=dict(
        step=1,
        non_grad_step=0,),
    test_cfg=dict(
        rollout=True,),
)