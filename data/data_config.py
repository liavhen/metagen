import sys
# sys.path.append('.')
# sys.path.append('..')
from edm_utils.dnnlib import EasyDict
from os.path import join
from utils.paths import DATA_DIR, ASSETS_DIR
from copy import copy
from dataclasses import dataclass

def edit_cfg(cfg, updates):
    cfg = copy(cfg)
    cfg.__dict__.update(updates)
    return cfg

@dataclass
class DataConfig:
    name: str
    substrate: str
    structure: str
    periodicity: float
    wavelengths: list
    heights: list
    lmdb_root_path: EasyDict
    ckpts_path: EasyDict
    info_t_orders: int = 19
    info_r_orders: int = 19
    roi_t_orders: int = 5
    roi_r_orders: int = 9
    use_t_only: bool = False
    use_r_only: bool = False
    use_te_only: bool = False
    use_tm_only: bool = False
    mean_total_t: float = 0.9
    resolution: int = 64
    rcwa_orders: int = 7

    def __repr__(self):
        return str(EasyDict(self.__dict__))
        # return f'DataConfig(name={self.name}):\n' + '\n'.join(f'(*) {k:15s}: {v}' for k, v in self.__dict__.items() if k != 'name')



# --------------------------------------------------------------------------------
# SiO2 - only freeform structures
# --------------------------------------------------------------------------------

cfg_a1 = DataConfig(
    name='a1',
    substrate = 'SiO2',
    structure = 'SiO2',
    info_t_orders = 3,
    info_r_orders = 5,
    roi_t_orders = 3,
    roi_r_orders = 5,
    use_t_only = True,
    use_r_only = False,
    use_te_only = True,
    use_tm_only = False,
    mean_total_t = 0.92, 
    periodicity = 1.9, # um
    resolution= 64, # pixels
    rcwa_orders = 7, 
    wavelengths = ['0.850', '0.900', '0.950', '1.000', '1.050', '1.100'], # um
    heights = [0.75, 1.45], # um
    lmdb_root_path=EasyDict(
        train   = join(DATA_DIR, 'metagen_a1'),
        test    = join(DATA_DIR, 'metagen_a1_test')
    ),
    ckpts_path=EasyDict(
        metagen = join(ASSETS_DIR, 'metagen',   'metagen-a1.pth'),
        cwgan   = join(ASSETS_DIR, 'cwgan',     'cwgan-a1.pth'),
        cvae    = join(ASSETS_DIR, 'cvae',      'cvae-a1.ckpt'),
    )
)

cfg_a2 = DataConfig(
    name='a2',
    substrate = 'SiO2',
    structure = 'SiO2',
    info_t_orders = 7,
    info_r_orders = 19,
    roi_t_orders = 5,
    roi_r_orders = 19,
    use_t_only = True,
    use_r_only = False,
    use_te_only = True,
    use_tm_only = False,
    mean_total_t = 0.94, # empiric total desired transmission calculated on a sampled batch by (max + average) / 2 
    periodicity = 3.2, # um
    resolution= 64, # pixels
    rcwa_orders = 7,
    wavelengths = ['0.850', '0.900', '0.950', '1.000', '1.050', '1.100'], # um
    heights = [0.75, 1.45], # um
    lmdb_root_path=EasyDict(
        train   = join(DATA_DIR, 'metagen_a2'),
        test    = join(DATA_DIR, 'metagen_a2_test')
    ),
    ckpts_path=EasyDict(
        metagen = join(ASSETS_DIR, 'metagen',   'metagen-a2.pth'),
        cwgan   = join(ASSETS_DIR, 'cwgan',     'cwgan-a2.pth'),
        cvae    = join(ASSETS_DIR, 'cvae',      'cvae-a2.ckpt'),
    )
)

cfg_a3 = DataConfig(
    name='a3',
    substrate = 'SiO2',
    structure = 'SiO2',
    info_t_orders = 11,
    info_r_orders = 15,
    roi_t_orders = 7,
    roi_r_orders = 9,
    use_t_only = True,
    use_r_only = False,
    use_te_only = True,
    use_tm_only = False,
    mean_total_t = 0.71, 
    periodicity = 4.6, # um
    resolution= 64, # pixels
    rcwa_orders = 7,
    wavelengths = ['0.850', '0.900', '0.950', '1.000', '1.050', '1.100'], # um
    # heights = ['1.000', '1.250', '1.500', '1.750', '2.000', '2.250', '2.500', '3.000'], # um
    heights = [0.75, 1.45], # um
    lmdb_root_path=EasyDict(
        train = join(DATA_DIR, 'metagen_a3'),
        test  = join(DATA_DIR, 'metagen_a3_test'),
    ),
    ckpts_path=EasyDict(
        metagen = join(ASSETS_DIR, 'metagen',   'metagen-a3.pth'),
        cwgan   = join(ASSETS_DIR, 'cwgan',     'cwgan-a3.pth'),
        cvae    = join(ASSETS_DIR, 'cvae',      'cvae-a3.ckpt'),
    )
)

cfg_a3_7x7 = edit_cfg(cfg_a3, {'roi_t_orders': 5})

# --------------------------------------------------------------------------------
# SiO2 - including structures from other families
# --------------------------------------------------------------------------------

cfg_b2 = DataConfig(
    name='b2',
    substrate = 'SiO2',
    structure = 'SiO2',
    info_t_orders = 7,
    info_r_orders = 19,
    roi_t_orders = 5,
    roi_r_orders = 19,
    use_t_only = True,
    use_r_only = False,
    use_te_only = True,
    use_tm_only = False,
    mean_total_t = 0.93, # empiric total desired transmission calculated on a sampled batch by (max + average) / 2 
    periodicity = 2.86, # um
    resolution= 64, # pixels
    rcwa_orders = 7,
    wavelengths = ['0.800', '0.850', '0.900', '0.950', '1.000'], # um
    heights = [0.75, 1.25], # um
    lmdb_root_path=EasyDict(
        train   = join(DATA_DIR, 'metagen_b2'),
        test    = join(DATA_DIR, 'metagen_b2_test')
    ),
    ckpts_path=EasyDict(
        metagen = join(ASSETS_DIR, 'metagen',   'metagen-b2.pth'),
        # cwgan   = join(ASSETS_DIR, 'cwgan',     'cwgan-b2.pth'),
        # cvae    = join(ASSETS_DIR, 'cvae',      'cvae-b2.ckpt'),
    )
)


# --------------------------------------------------------------------------------
# Silicon
# --------------------------------------------------------------------------------

cfg_c2 = DataConfig(
    name='c2',
    substrate = 'Si',
    structure = 'Si',
    info_t_orders = 7,
    info_r_orders = 19,
    roi_t_orders = 5,
    roi_r_orders = 19,
    use_t_only = True,
    use_r_only = False,
    use_te_only = False,
    use_tm_only = False,
    mean_total_t = 0.5818, # empiric total desired transmission calculated on a sampled batch by (max + average) / 2 
    periodicity = 4.6, # um
    resolution= 64, # pixels
    rcwa_orders = 9,
    wavelengths = ['1.400', '1.450', '1.500', '1.550', '1.600'], # um
    heights = [0.1, 0.6], # um
    lmdb_root_path=EasyDict(
        train = join(DATA_DIR, 'metagen_c2'),
        test  = join(DATA_DIR, 'metagen_c2_test'),
    ),
    ckpts_path=EasyDict(
        metagen = join(ASSETS_DIR, 'metagen',   'metagen-c2.pth'),
        # cwgan   = join(ASSETS_DIR, 'cwgan',     'cwgan-c2.pth'),
        # cvae    = join(ASSETS_DIR, 'cvae',      'cvae-c2.ckpt'),
    )
)



def get_data_cfg(cfg_name):
    if cfg_name == 'a1':
        return cfg_a1
    elif cfg_name == 'a2':
        return cfg_a2
    elif cfg_name == 'a3':
        return cfg_a3
    elif cfg_name == 'b2':
        return cfg_b2
    elif cfg_name == 'c2':
        return cfg_c2
    else:
        raise ValueError(f'Unknown configuration: {cfg_name}')

