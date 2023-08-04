_base_ = ['./ep1.py']

# Custom model
model = dict(
    decode_head=dict(
        accuracy=[
            dict(type='CollisionAccuracy'),
            dict(type='L2Accuracy', reduction='mean'),],),)
# Custom dataset
data = dict(
    test=dict(
        env_cfg=dict(
            # merged=True,
            collision=True,),),)