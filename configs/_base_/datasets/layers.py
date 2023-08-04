env_cfg = dict(
	layers_base=dict(
		root_dir = 'data/',
		generated_dir = 'data/generated_data',
		smpl_dir = 'data/smpl/model',
		split_meta = 'data/data/train_val_test.json',
		train_split = 0.8,
		val_split = 0.1,
		test_split = 0.1,
		dt=1.0, # Only for computing, the real one is controled by fps
		radius=0.4,
		human_radius=0.6,
		fps=30, # Only for calculate the gravity
		),
)

data = dict(
	train=dict(
		phase='train',
		env_cfg=env_cfg,),
	val=dict(
		phase='valid',
		env_cfg=env_cfg,),
	test=dict(
		phase='valid', 
		env_cfg=env_cfg,),
)