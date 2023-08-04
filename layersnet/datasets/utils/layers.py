from itertools import chain
import os
import numpy as np
from pickle import UnpicklingError
from layersnet.datasets.utils import readJSON, readPKL, patchwise, writePKL
from layersnet.datasets.smpl import SMPLModel


META_FN = 'meta.json'

GARMENT_TYPE = [
    'jacket', 
    'jacket_hood',
    'jumpsuit_sleeveless', 
    'tee', 
    'tee_sleeveless',
    'dress_sleeveless', 
    'wb_dress_sleeveless',
    'wb_pants_straight', 
    'pants_straight_sides',
    'skirt_2_panels', 
    'skirt_4_panels', 
    'skirt_8_panels',
]

COLLATE_GARMENT_TYPE = [
    'jacket',
    'jumpsuit_sleeveless',
    'tee',
    'dress_sleeveless',
    'pants_straight',
    'skirt',
]

ATTR_RANGE = dict(
    tension=(15, 100),
    bending=(15, 200),
    mass=(0.2, 0.8),
    friction=(40, 80),
)

WIND_RANGE = dict(
    strength=(0, 400)
)

class LayersReader:
    def __init__(self, layers_cfg, phase='train'):
        self.cfg = layers_cfg
        self.phase = phase
        self.data_dir = layers_cfg.root_dir
        self.generated_dir = layers_cfg.generated_dir
        self.seq_list = []
        split_meta = readJSON(layers_cfg.split_meta)
        if isinstance(phase, list):
            phase_meta = list(chain(*[split_meta[i] for i in phase]))
        else:
            phase_meta = split_meta[phase]
        for seq_num in phase_meta:
            self.seq_list.append(seq_num)

        self.smpl = {
			'female': SMPLModel(os.path.join(self.cfg.smpl_dir, 'model_f.pkl')),
			'male': SMPLModel(os.path.join(self.cfg.smpl_dir, 'model_m.pkl'))
		}

        self.garment_type = dict()
        one_hot = np.eye(len(GARMENT_TYPE))
        for g_name, g_code in zip(GARMENT_TYPE, one_hot):
            self.garment_type[g_name] = g_code

    """ 
	Read sample info 
	Input:
	- sample: name of the sample e.g.:'01_01_s0'
	"""
    def read_info(self, sample):
        info_path = os.path.join(self.data_dir, 'data', sample, META_FN)
        infos = readJSON(info_path)
        # Add one hot coding
        for i in range(len(infos['garment'])):
            infos['garment'][i]['type'] = self.garment_type[infos['garment'][i]['name']]
        return infos
        
    """ Human data """
    """
	Read SMPL parameters for the specified sample and frame
	Inputs:
	- sample: name of the sample
	- frame: frame number
	"""
    def read_smpl_params(self, sample, frame):
		# Read sample data
        info = self.read_info(sample)
		# SMPL parameters
        gender = info['human']['gender']
        motion = readPKL(os.path.join(self.data_dir, info['human']['motion_path']))
        pose = motion['poses']
        trans = motion['trans']
        frame += info['human']['seq_start']
        if len(pose) == 1: frame = None
        pose = pose[frame].reshape(self.smpl[gender].pose_shape)
        shape = np.array(info['human']['betas'])
        # since when generating, the scale is not applied to the trans
        trans = trans[frame].reshape(self.smpl[gender].trans_shape) * info['human']['scale']
        return gender, pose, shape, trans
	
    """
	Computes human mesh for the specified sample and frame
	Inputs:
	- sample: name of the sample
	- frame: frame number
	Outputs:
	- V: human mesh vertices
	- F: mesh faces
	"""
    def read_human(self, sample, frame):
		# Read sample data
        info = self.read_info(sample)
        assert frame is not None
        gender, pose, shape, trans = self.read_smpl_params(sample, frame)
        # Compute SMPL
        _, V, root_offset = self.smpl[gender].set_params(pose=pose, beta=shape, trans=trans/info['human']['scale'], with_body=True)
        root_offset *= info['human']['scale']
        V *= info['human']['scale']
        F = self.smpl[gender].faces.copy()
        V += root_offset
        return V, F
	
    """ Garment data """
    """
	Reads garment vertices location for the specified sample, garment and frame
	Inputs:
	- sample: name of the sample
	- garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
	- frame: frame number
	- absolute: True for absolute vertex locations, False for locations relative to SMPL root joint	
	Outputs:
	- V: 3D vertex locations for the specified sample, garment and frame
	"""
    def read_garment_vertices(self, sample, garment, frame=None):
		# Read garment vertices (relative to root joint)
        garment_path = os.path.join(self.data_dir, 'data', sample, garment + '.pkl')
        if not os.path.exists(garment_path):
            return None
        garment_seq = readPKL(garment_path)
        V = garment_seq['vertices']
        if frame is not None:
            V = garment_seq['vertices'][frame]
        return V
    
    """ Garment attr """
    """
	Reads garment vertices location for the specified sample, garment and frame
	"""
    def read_garment_attributes(self, sample, garment, info=None):
		# Read garment vertices (relative to root joint)
        if info is None:
            info = self.read_info(sample)
        attr = None
        for g_meta in info['garment']:
            if g_meta['name'] == garment:
                attr = np.array([
                (g_meta['cloth']['tension_stiffness'] - ATTR_RANGE['tension'][0]) / (ATTR_RANGE['tension'][1]-ATTR_RANGE['tension'][0]),
                (g_meta['cloth']['bending_stiffness'] - ATTR_RANGE['bending'][0]) / (ATTR_RANGE['bending'][1]-ATTR_RANGE['bending'][0]),
                (g_meta['cloth']['mass'] - ATTR_RANGE['mass'][0]) / (ATTR_RANGE['mass'][1] - ATTR_RANGE['mass'][0]),
                (g_meta['cloth']['self_friction'] - ATTR_RANGE['friction'][0]) / (ATTR_RANGE['friction'][1] - ATTR_RANGE['friction'][0]),
                (info['human']['collision']['friction'] - ATTR_RANGE['friction'][0]) / (ATTR_RANGE['friction'][1] - ATTR_RANGE['friction'][0]),
                ])
                break
        assert attr is not None
        return attr

    """
	Reads garment faces for the specified sample and garment
	"""
    def read_garment_topology(self, sample, garment):
		# Read OBJ file
        pkl_path = os.path.join(self.data_dir, 'data', sample, garment + '.pkl')
        garment_seq = readPKL(pkl_path)
        F = garment_seq['faces']
        T = garment_seq['tpose'] # T-pose templates
        return F, T
    
    """ Garment data """
    """
	Reads garment vertices location for the specified sample, garment and frame
    and the faces
	"""
    def read_garment_vertices_topology(self, sample, garment, frame):
		# Read garment vertices (relative to root joint)
        garment_path = os.path.join(self.data_dir, 'data', sample, garment + '.pkl')
        garment_seq = readPKL(garment_path)
        V = garment_seq['vertices'][frame]
        F = garment_seq['faces']
        T = garment_seq['tpose']
        return V, F, T

    """	
	Reads garment UV map for the specified sample and garment
	Inputs:
	- sample: name of the sample
	- garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
	Outputs:
	- Vt: UV map vertices
	- Ft: UV map faces		
	"""
    def read_garment_UVMap(self, sample, garment):
        uv_path = os.path.join(self.generated_dir, sample, f"uv_{garment}.pkl")
        uv_groups = readPKL(uv_path)
        return uv_groups

    """	
	Reads patched garment masks and faces groups
	Inputs:
	- sample: name of the sample
	- garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
	Outputs:
	- Vt: UV map vertices
	- Ft: UV map faces		
	"""
    def read_patched_garment(self, sample, garment, patch_size=0.1):
        patch_dir = os.path.join(self.data_dir, 'data', sample)
        patch_path = os.path.join(patch_dir, f"patch_{garment}_{patch_size}".replace('.', 'd') + ".pkl")
        try:
            data = readPKL(patch_path)
        except (UnpicklingError, FileNotFoundError, EOFError):
            uv_path = os.path.join(self.data_dir, 'data', sample, f"uv_{garment}.pkl")
            uv_groups = readPKL(uv_path)
            mask_vr, mask_vc, mask_fr, mask_fc, patch_offset = patchwise(uv_groups, p_size=patch_size)
            ft_list = [uv['faces'] for vg_name, uv in uv_groups.items()]
            data = dict(
                # This is the global idx
                v2p_mask=dict(
                    row=mask_vr,
                    col=mask_vc,
                ),
                f2p_mask=dict(
                    row=mask_fr,
                    col=mask_fc,
                ),
                faces=ft_list
            )
            writePKL(patch_path, data)
        return data		
	
    def read_wind(self, sample, frame, info=None):
		# Read garment vertices (relative to root joint)
        if info is None:
            info = self.read_info(sample)
        wind_info = info['simulate']['wind']
        seq_length = info['simulate']['human']['seq_end'] - info['simulate']['human']['seq_start']
        outer_forces_idx = [
                o['frame_start'] - info['simulate']['human']['frame_start']
                for o in wind_info[0]['pivot_list']] \
                    + [seq_length]

        if frame < outer_forces_idx[0]:
            w_info = np.zeros(5)
        else:
            for o_idx in range(1, len(outer_forces_idx)):
                assert outer_forces_idx[o_idx-1] <= outer_forces_idx[o_idx]
                if frame < outer_forces_idx[o_idx] and frame >= outer_forces_idx[o_idx-1]:
                    # find it
                    w_meta = wind_info[0]['pivot_list'][o_idx-1]
                    rotations = w_meta['rotation_quaternion']
                    strengths = np.array(w_meta['strength']).reshape(1)
                    w_info = np.concatenate([rotations, strengths], axis=0)
                    break
        return w_info
    
    def normalize_wind(self, wind_info):
        wind_info[:, -1] = (wind_info[:, -1] - WIND_RANGE['strength'][0]) / (WIND_RANGE['strength'][1] - WIND_RANGE['strength'][0])
        return wind_info