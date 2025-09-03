import sys
sys.path.append('.')
sys.path.append('..')
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from os.path import join
from utils.paths import *
from utils import utils
import numpy as np

"""
This script is used to test the meta-atoms generations based on special scattering patterns:
1. Beam-Splitter (uniform and efficient transmittance)
2. Prismatic transmittance (Maximizing 1st order diffraction)
3. Dilated transmittance (all order but (0,0) are efficient)
5. Horizontal spread (both x=1, -1, orders are efficient)
4. Vertical spread (both y=1, -1 orders are efficient)
"""


def get_uniform_diffraction(t=3, r=5, n=19, desired_total_t=0.9):
    """
    :param t: number of diffraction orders in transmittance
    :param r: number of diffraction orders in reflection
    :param n: resolution of scattering output image
    :return: T, R
    """
    total_t = desired_total_t
    total_r = 1.0 - total_t
    t_pad = ((n-t)//2,)*4
    r_pad = ((n-r)//2,)*4
    T = F.pad(total_t / t**2 * torch.ones((t, t)), pad=t_pad)
    # R = F.pad(total_r / r**2 * torch.ones((r, r)), pad=r_pad) if r != 0 else torch.zeros((n, n))
    R = torch.zeros_like(T)
    R[n//2, n//2] = total_r
    return T, R


def get_prismatic_diffraction(order=(1, 1), t=3, r=5, n=19, desired_total_t=0.9):
    """
    :param order: diffraction order with concentrated energy [(x,y) format)
    :param t: number of diffraction orders in transmittance
    :param r: number of diffraction orders in reflection
    :param n: resolution of scattering output image
    :return: T, R
    """
    assert order[0] <= t//2 and order[1] <= t//2, "Illegal diffraction order in the current physical settings!"
    total_t = desired_total_t
    total_r = 1.0 - total_t
    T = torch.zeros((n, n))
    T[n//2 - order[1], n//2 + order[0]] = total_t
    r_pad = ((n-r)//2,)*4
    # R = F.pad(total_r / r**2 * torch.ones((r, r)), pad=r_pad) if r != 0 else torch.zeros((n, n))
    R = torch.zeros_like(T)
    R[n//2, n//2] = total_r
    return T, R


def get_multi_prismatic_diffraction(orders=None, t=3, r=5, n=19, desired_total_t=0.9):
    if orders is None:
        orders = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    T = torch.zeros((n, n))
    R = torch.zeros((n, n))
    for o in orders:
        _T, _R = get_prismatic_diffraction(order=o, t=t, r=r, desired_total_t=desired_total_t)
        T += _T / len(orders)
        R += _R / len(orders)
    return T, R


def flatten_conditions(scattering, resolution=19, r=5, t=3):
    T, R = scattering
    return utils.flatten_conditions(T, R, resolution=resolution, r_nof_angles=r, t_nof_angles=t)


def get_gaussian(n, sigma, mask_radius=None):
    """
    Create an n x n matrix with a Gaussian distribution centered at the middle.
    
    :param n: Size of the matrix (must be odd)
    :param std_dev: Standard deviation of the Gaussian
    :return: n x n tensor with Gaussian distribution
    """
    assert n % 2 == 1, "n must be odd"
    
    # Create a grid of (x, y) coordinates
    x = torch.arange(n, device=torch.device('cuda')) - n // 2
    y = torch.arange(n, device=torch.device('cuda')) - n // 2
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    
    # Calculate the Gaussian
    gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    
    if mask_radius is not None:
        mask = (x_grid**2 + y_grid**2) <= mask_radius**2
        gaussian *= mask

    # Normalize the Gaussian to have a sum of 1
    gaussian /= gaussian.sum()
    
    return gaussian


def get_base_target_patterns(data_cfg, add_noise=0):
    """
    Returns a dictionary with several special scattering T,R patterns, each flattened into a 2*19**2-dimensional array (flattened 19x19 for each {T,R}).
    This does not include the wavelength which is also guiding the diffusion model,
    thus must be added before entering the model (see `get_special_conditions()` function below).
    """
    
    ti, ri = data_cfg.info_t_orders, data_cfg.info_r_orders
    t, r = data_cfg.roi_t_orders, data_cfg.roi_r_orders
    total_t = data_cfg.mean_total_t
    res = {
        'uniform':                      get_uniform_diffraction(t=t, r=r, desired_total_t=total_t),
        'prism_1_0':                    get_prismatic_diffraction(order=( 1, 0), t=t, r=r, desired_total_t=total_t),
        'prism_-1_0':                   get_prismatic_diffraction(order=(-1, 0), t=t, r=r, desired_total_t=total_t),
        'prism_0_1':                    get_prismatic_diffraction(order=( 0, 1), t=t, r=r, desired_total_t=total_t),
        'prism_0_-1':                   get_prismatic_diffraction(order=( 0,-1), t=t, r=r, desired_total_t=total_t),
        'horizontal_1st_orderes':       get_multi_prismatic_diffraction(orders=[(1, 0), (-1,0)], t=t, r=r, desired_total_t=total_t),
        'vertical_1st_orderes':         get_multi_prismatic_diffraction(orders=[(0, 1), (0,-1)], t=t, r=r, desired_total_t=total_t),
        'horizontal_full':              get_multi_prismatic_diffraction(orders=[(x, 0) for x in range(-t//2 + 1, t//2 + 1)], t=t, r=r, desired_total_t=total_t),
        'horizontal_dilated':           get_multi_prismatic_diffraction(orders=[(x, 0) for x in range(-t//2 + 1, t//2 + 1) if x != 0], t=t, r=r, desired_total_t=total_t),
        'vertical_full':                get_multi_prismatic_diffraction(orders=[(0, y) for y in range(-t//2 + 1, t//2 + 1)], t=t, r=r, desired_total_t=total_t),
        'vertical_dilated':             get_multi_prismatic_diffraction(orders=[(0, y) for y in range(-t//2 + 1, t//2 + 1) if y != 0], t=t, r=r, desired_total_t=total_t),
        'big_cross':                    get_multi_prismatic_diffraction(orders=[(x, y) for x in range(-ti//2 + 1, ti//2 + 1) for y in range(-ti//2 + 1, ti//2 + 1) if x==0 or y==0], t=ti, r=ri, desired_total_t=total_t),
        'small_cross':                  get_multi_prismatic_diffraction(orders=[(x, y) for x in range(-t//2 + 1, t//2 + 1) for y in range(-t//2 + 1, t//2 + 1) if x==0 or y==0], t=t, r=r, desired_total_t=total_t),
        'vertical_dilated_stripes':     get_multi_prismatic_diffraction(orders=[(x, y) for x in range(-t//2 + 1, t//2 + 1) for y in range(-t//2 + 1, t//2 + 1) if x%2==0], t=t, r=r, desired_total_t=total_t),
        'horizontal_dilated_stripes':   get_multi_prismatic_diffraction(orders=[(x, y) for x in range(-t//2 + 1, t//2 + 1) for y in range(-t//2 + 1, t//2 + 1) if y%2==0], t=t, r=r, desired_total_t=total_t),
        'odd_dilated':                  get_multi_prismatic_diffraction(orders=[(x, y) for x in range(-t//2 + 1, t//2 + 1) for y in range(-t//2 + 1, t//2 + 1) if x%2==1 and y%2==1], t=t, r=r, desired_total_t=total_t),
        'frontslash':                   get_multi_prismatic_diffraction(orders=[(x, y) for x in range(-t//2 + 1, t//2 + 1) for y in range(-t//2 + 1, t//2 + 1) if x==y], t=t, r=r, desired_total_t=total_t),
        'backslash':                    get_multi_prismatic_diffraction(orders=[(x, y) for x in range(-t//2 + 1, t//2 + 1) for y in range(-t//2 + 1, t//2 + 1) if x==-y], t=t, r=r, desired_total_t=total_t),
        'x':                            get_multi_prismatic_diffraction(orders=[(x, y) for x in range(-t//2 + 1, t//2 + 1) for y in range(-t//2 + 1, t//2 + 1) if x==y or x==-y], t=t, r=r, desired_total_t=total_t),
    }

    if data_cfg.roi_t_orders > 3:
        res.update({
            'prism_2_0':                    get_prismatic_diffraction(order=( 2, 0), t=t, r=r, desired_total_t=total_t),
            'prism_0_2':                    get_prismatic_diffraction(order=( 0, 2), t=t, r=r, desired_total_t=total_t),
            'even_dilated':                 get_multi_prismatic_diffraction(orders=[(x, y) for x in range(-t//2 + 1, t//2 + 1) for y in range(-t//2 + 1, t//2 + 1) if x%2==0 and y%2==0], t=t, r=r, desired_total_t=total_t),
        })

    if data_cfg.roi_t_orders > 5:
        res.update({
            'prism_3_0':                    get_prismatic_diffraction(order=( 3, 0), t=ti,r=ri, desired_total_t=total_t),
            'prism_0_3':                    get_prismatic_diffraction(order=( 0, 3), t=ti,r=ri, desired_total_t=total_t),
        })

    if add_noise > 0:
        Rt = data_cfg.periodicity / np.array([float(l) for l in data_cfg.wavelengths]).mean()
        for k,v in res.items():
            _t, _r = v
            _t, _r = _t.cuda(), _r.cuda()
            g = get_gaussian(_t.shape[0], sigma=100, mask_radius=Rt)
            _t = _t + add_noise * torch.rand_like(_t) * g
            _t = _t.clip(min=0)  # ensure that adding the noise doesn't result in negative values
            _t /= (_t + _r).sum()
            _r /= (_t + _r).sum()
            res[k] = (_t, _r)

    for k, v, in res.items():
        # res[k] = flatten_conditions(v, t=data_cfg.info_t_orders, r=data_cfg.info_r_orders).cuda()
        t, r = v
        use_t, use_r, _, _ = utils.get_component_types_booleans(data_cfg)
        included_components = use_t * [utils.crop_around_center(t, data_cfg.info_t_orders).flatten()] + use_r * [utils.crop_around_center(r, data_cfg.info_r_orders).flatten()]
        res[k] = torch.cat(included_components).cuda()
    return res


def get_base_polarized_target_patterns(data_cfg, add_noise=0):
    base_scatterings = get_base_target_patterns(data_cfg, add_noise=add_noise)
    return {k: torch.cat([v, v]) for k, v in base_scatterings.items()}

def get_polarized_target_patterns(data_cfg, add_noise=0):
    """ 
    Used to assemble special scatterings for the dual-polarized model, i.e - 
    generate a joint scattering requirement for both TE and TM polarizations.
    """

    base_scatterings = get_base_target_patterns(data_cfg, add_noise=add_noise)
    base_polarized_target_patters = get_base_polarized_target_patterns(data_cfg, add_noise=add_noise)
    
    # naming convention: p_ prefix denotes TE and TM are different
    te_tm_scatterings = {                           #                        TE                                             TM
        # 'p_invariant_uniform':                      torch.cat([base_scatterings['uniform'],                     base_scatterings['uniform']]),
        'p_splitter_vertical_1':         torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_0_-1']]),
        'p_splitter_vertical_2':         torch.cat([base_scatterings['prism_0_-1'],                  base_scatterings['prism_0_1']]),
        'p_splitter_horizontal_1':       torch.cat([base_scatterings['prism_1_0'],                   base_scatterings['prism_-1_0']]),
        'p_splitter_horizontal_2':       torch.cat([base_scatterings['prism_-1_0'],                  base_scatterings['prism_1_0']]),

        # 'p_prism_1_0':                              torch.cat([base_scatterings['prism_1_0'],                   base_scatterings['prism_1_0']]),
        # 'p_prism_-1_0':                             torch.cat([base_scatterings['prism_-1_0'],                  base_scatterings['prism_-1_0']]),
        # 'p_prism_0_1':                              torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_0_1']]),
        # 'p_prism_0_-1':                             torch.cat([base_scatterings['prism_0_-1'],                  base_scatterings['prism_0_-1']]),
        # 'horizontal_deflector_te_only':             torch.cat([base_scatterings['prism_1_0'],                   base_scatterings['prism_0_-1']*0]),
        # 'horizontal_deflector_tm_only':             torch.cat([base_scatterings['prism_1_0']*0,                 base_scatterings['prism_1_0']]),
        # 'vertical_deflector_te_only':               torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_0_-1']*0]),
        # 'vertical_deflector_tm_only':               torch.cat([base_scatterings['prism_0_1']*0,                 base_scatterings['prism_0_1']]),
        # 'p_slashes_1':                              torch.cat([base_scatterings['frontslash'],                  base_scatterings['frontslash']]),
        # 'p_slashes_2':                              torch.cat([base_scatterings['frontslash'],                  base_scatterings['backslash']]),
        # 'p_slashes_3':                              torch.cat([base_scatterings['backslash'],                   base_scatterings['frontslash']]),
        # 'p_invariant_x':                            torch.cat([base_scatterings['x'],                           base_scatterings['x']]),
        # 'polarization_stripes_splitter_1':          torch.cat([base_scatterings['horizontal_dilated_stripes'],  base_scatterings['vertical_dilated_stripes']]),
        # 'polarization_stripes_splitter_2':          torch.cat([base_scatterings['vertical_dilated_stripes'],    base_scatterings['horizontal_dilated_stripes']]),
    }
    output_scatterings = {**base_polarized_target_patters, **te_tm_scatterings}
    return output_scatterings


def get_polarized_beam_deflectors(data_cfg, add_noise=0):
    """ 
    Used to assemble special scatterings for the dual-polarized model, i.e - 
    generate a joint scattering requirement for both TE and TM polarizations.
    """

    base_scatterings = get_base_target_patterns(data_cfg, add_noise=add_noise)

    prism_scatterings = {                  #                        TE                                             TM
        
        # Single order (axes aligned) single polarization deflection
        'te_1_0':                             torch.cat([base_scatterings['prism_1_0'],                   base_scatterings['prism_0_-1']*0]),
        'te_0_1':                             torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_0_-1']*0]),
        'te_0_-1':                            torch.cat([base_scatterings['prism_0_-1'],                  base_scatterings['prism_0_1']*0]),
        'te_-1_0':                            torch.cat([base_scatterings['prism_-1_0'],                  base_scatterings['prism_0_1']*0]),
        'tm_1_0':                             torch.cat([base_scatterings['prism_1_0']*0,                 base_scatterings['prism_1_0']]),
        'tm_0_1':                             torch.cat([base_scatterings['prism_0_1']*0,                 base_scatterings['prism_0_1']]),
        'tm_0_-1':                            torch.cat([base_scatterings['prism_0_-1']*0,                base_scatterings['prism_0_-1']]),
        'tm_-1_0':                            torch.cat([base_scatterings['prism_-1_0']*0,                base_scatterings['prism_-1_0']]),
        
        'te_2_0':                             torch.cat([base_scatterings['prism_2_0'],                   base_scatterings['prism_0_-1']*0]),
        'te_0_2':                             torch.cat([base_scatterings['prism_0_2'],                   base_scatterings['prism_0_-1']*0]),
        'te_0_-2':                            torch.cat([base_scatterings['prism_0_-2'],                  base_scatterings['prism_0_1']*0]),
        'te_-2_0':                            torch.cat([base_scatterings['prism_-2_0'],                  base_scatterings['prism_0_1']*0]),
        'tm_2_0':                             torch.cat([base_scatterings['prism_2_0']*0,                 base_scatterings['prism_2_0']]),
        'tm_0_2':                             torch.cat([base_scatterings['prism_0_2']*0,                 base_scatterings['prism_0_2']]),
        'tm_0_-2':                            torch.cat([base_scatterings['prism_0_-2']*0,                base_scatterings['prism_0_-2']]),
        'tm_-2_0':                            torch.cat([base_scatterings['prism_-2_0']*0,                base_scatterings['prism_-2_0']]),

        # Single order (axes an-aligned), single polarization deflection
        'te_1_1':                             torch.cat([base_scatterings['prism_1_1'],                   base_scatterings['prism_0_-1']*0]),
        'te_1_-1':                            torch.cat([base_scatterings['prism_1_-1'],                  base_scatterings['prism_0_1']*0]),
        'te_-1_1':                            torch.cat([base_scatterings['prism_-1_1'],                  base_scatterings['prism_0_-1']*0]),
        'te_-1_-1':                           torch.cat([base_scatterings['prism_-1_-1'],                 base_scatterings['prism_0_1']*0]),
        'tm_1_1':                             torch.cat([base_scatterings['prism_1_1']*0,                 base_scatterings['prism_1_1']]),
        'tm_1_-1':                            torch.cat([base_scatterings['prism_1_-1']*0,                base_scatterings['prism_1_-1']]),
        'tm_-1_1':                            torch.cat([base_scatterings['prism_-1_1']*0,                base_scatterings['prism_-1_1']]),
        'tm_-1_-1':                           torch.cat([base_scatterings['prism_-1_-1']*0,               base_scatterings['prism_-1_-1']]),

        # Single order, double polarization deflection
        ## Matching TE and TM
        'te_1_0_tm_1_0':                      torch.cat([base_scatterings['prism_1_0'],                   base_scatterings['prism_1_0']]),
        'te_0_1_tm_0_1':                      torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_0_1']]),
        'te_0_-1_tm_0_-1':                    torch.cat([base_scatterings['prism_0_-1'],                  base_scatterings['prism_0_-1']]),
        'te_-1_0_tm_-1_0':                    torch.cat([base_scatterings['prism_-1_0'],                  base_scatterings['prism_-1_0']]),
        ## Non-matching TE and TM
        'te_1_0_tm_0_1':                      torch.cat([base_scatterings['prism_1_0'],                   base_scatterings['prism_0_1']]),
        'te_0_1_tm_1_0':                      torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_1_0']]),
        'te_0_1_tm_1_0':                      torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_1_0']]),
        'te_1_0_tm_2_0':                      torch.cat([base_scatterings['prism_1_0'],                   base_scatterings['prism_2_0']]),
        'te_0_1_tm_0_2':                      torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_0_2']]),

        # Double order, single polarization deflection
        # Symmetric
        'te_h_1st':                           torch.cat([base_scatterings['horizontal_1st_orderes'],     base_scatterings['horizontal_1st_orderes']*0]),
        'te_h_2nd':                           torch.cat([base_scatterings['horizontal_2nd_orderes'],     base_scatterings['horizontal_2nd_orderes']*0]),
        'te_v_1st':                           torch.cat([base_scatterings['vertical_1st_orderes'],       base_scatterings['vertical_1st_orderes']*0]),
        'te_v_2nd':                           torch.cat([base_scatterings['vertical_2nd_orderes'],       base_scatterings['vertical_2nd_orderes']*0]),
        'tm_h_1st':                           torch.cat([base_scatterings['horizontal_1st_orderes']*0,   base_scatterings['horizontal_1st_orderes']]),
        'tm_h_2nd':                           torch.cat([base_scatterings['horizontal_2nd_orderes']*0,   base_scatterings['horizontal_2nd_orderes']]),
        'tm_v_1st':                           torch.cat([base_scatterings['vertical_1st_orderes']*0,     base_scatterings['vertical_1st_orderes']]),
        'tm_v_2nd':                           torch.cat([base_scatterings['vertical_2nd_orderes']*0,     base_scatterings['vertical_2nd_orderes']]),
        # Asymmetric
        'te_h_-1_2':                          torch.cat([base_scatterings['horizontal_-1_2_orders'],    base_scatterings['horizontal_-1_2_orders']*0]),
        'te_h_-2_1':                          torch.cat([base_scatterings['horizontal_-2_1_orders'],    base_scatterings['horizontal_-2_1_orders']*0]),
        'te_v_-1_2':                          torch.cat([base_scatterings['vertical_-1_2_orders'],      base_scatterings['vertical_-1_2_orders']*0]),
        'te_v_-2_1':                          torch.cat([base_scatterings['vertical_-2_1_orders'],      base_scatterings['vertical_-2_1_orders']*0]),
        'tm_h_-1_2':                          torch.cat([base_scatterings['horizontal_-1_2_orders']*0,  base_scatterings['horizontal_-1_2_orders']]),
        'tm_h_-2_1':                          torch.cat([base_scatterings['horizontal_-2_1_orders']*0,  base_scatterings['horizontal_-2_1_orders']]),
        'tm_v_-1_2':                          torch.cat([base_scatterings['vertical_-1_2_orders']*0,    base_scatterings['vertical_-1_2_orders']]),
        'tm_v_-2_1':                          torch.cat([base_scatterings['vertical_-2_1_orders']*0,    base_scatterings['vertical_-2_1_orders']]),

        # Double order, double polarization deflection
        # Matching TE and TM
        'te_h_1st_tm_h_1st':                  torch.cat([base_scatterings['horizontal_1st_orderes'],     base_scatterings['horizontal_1st_orderes']]),
        'te_h_2nd_tm_h_2nd':                  torch.cat([base_scatterings['horizontal_2nd_orderes'],     base_scatterings['horizontal_2nd_orderes']]),
        'te_v_1st_tm_v_1st':                  torch.cat([base_scatterings['vertical_1st_orderes'],       base_scatterings['vertical_1st_orderes']]),
        'te_v_2nd_tm_v_2nd':                  torch.cat([base_scatterings['vertical_2nd_orderes'],       base_scatterings['vertical_2nd_orderes']]),
        # Non-matching TE and TM
        'te_h_1st_tm_v_1st':                  torch.cat([base_scatterings['horizontal_1st_orderes'],     base_scatterings['vertical_1st_orderes']]),
        'te_h_1st_tm_v_2nd':                  torch.cat([base_scatterings['horizontal_1st_orderes'],     base_scatterings['vertical_2nd_orderes']]),
        'te_h_2nd_tm_v_1st':                  torch.cat([base_scatterings['horizontal_2nd_orderes'],     base_scatterings['vertical_1st_orderes']]),
        'te_h_2nd_tm_v_2nd':                  torch.cat([base_scatterings['horizontal_2nd_orderes'],     base_scatterings['vertical_2nd_orderes']]),
        'te_v_1st_tm_h_1st':                  torch.cat([base_scatterings['vertical_1st_orderes'],       base_scatterings['horizontal_1st_orderes']]),
        'te_v_1st_tm_h_2nd':                  torch.cat([base_scatterings['vertical_1st_orderes'],       base_scatterings['horizontal_2nd_orderes']]),
        'te_v_2nd_tm_h_1st':                  torch.cat([base_scatterings['vertical_2nd_orderes'],       base_scatterings['horizontal_1st_orderes']]),
        'te_v_2nd_tm_h_2nd':                  torch.cat([base_scatterings['vertical_2nd_orderes'],       base_scatterings['horizontal_2nd_orderes']]),
        'te_h_1st_tm_h_2nd':                  torch.cat([base_scatterings['horizontal_1st_orderes'],     base_scatterings['horizontal_2nd_orderes']]),
        'te_h_2nd_tm_h_1st':                  torch.cat([base_scatterings['horizontal_2nd_orderes'],     base_scatterings['horizontal_1st_orderes']]),
        'te_v_1st_tm_v_2nd':                  torch.cat([base_scatterings['vertical_1st_orderes'],       base_scatterings['vertical_2nd_orderes']]),
        'te_v_2nd_tm_v_1st':                  torch.cat([base_scatterings['vertical_2nd_orderes'],       base_scatterings['vertical_1st_orderes']]),

    }
    

    return prism_scatterings


def get_polarized_desired_scattering_for_optimization(data_cfg, add_noise=0):
    """ 
    Used to assemble special scatterings for the dual-polarized model, i.e - 
    generate a joint scattering requirement for both TE and TM polarizations.
    This set is a collection of scatterings that are used for an optimization process.
    """

    base_scatterings = get_base_target_patterns(data_cfg, add_noise=add_noise)

    scatterings = {                           #                        TE                                                   TM
        
        # Dual polarizastion prisms
        'prism_te_1_0':                             torch.cat([base_scatterings['prism_1_0'],                   base_scatterings['prism_1_0']]),
        'prism_te_0_1':                             torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_1_0']]),
        'prism_te_0_-1':                            torch.cat([base_scatterings['prism_0_-1'],                  base_scatterings['prism_0_-1']]),
        'prism_te_-1_0':                            torch.cat([base_scatterings['prism_-1_0'],                  base_scatterings['prism_-1_0']]),
        
        # Single order polarization splitters
        'splitter_te_1_0_tm_-1_0':                  torch.cat([base_scatterings['prism_1_0'],                   base_scatterings['prism_-1_0']]),
        'splitter_te_-1_0_tm_1_0':                  torch.cat([base_scatterings['prism_-1_0'],                  base_scatterings['prism_1_0']]),
        'splitter_te_0_1_tm_0_-1':                  torch.cat([base_scatterings['prism_0_1'],                   base_scatterings['prism_0_-1']]),
        'splitter_te_0_-1_tm_0_1':                  torch.cat([base_scatterings['prism_0_-1'],                  base_scatterings['prism_0_1']]),
        
        # Double order polarization splitters
        'te_h_1st_tm_v_1st':                  torch.cat([base_scatterings['horizontal_1st_orderes'],     base_scatterings['vertical_1st_orderes']]),
        'te_v_1st_tm_h_1st':                  torch.cat([base_scatterings['vertical_1st_orderes'],       base_scatterings['horizontal_1st_orderes']]),
    
        # Uniform scattering
        'uniform':                                  torch.cat([base_scatterings['uniform'],                     base_scatterings['uniform']]),
        'te_uniform':                               torch.cat([base_scatterings['uniform'],                     base_scatterings['uniform']*0]),
        'tm_uniform':                               torch.cat([base_scatterings['uniform']*0,                   base_scatterings['uniform']]),
    }
    

    return scatterings


def get_special_conditions(data_cfg, device=torch.device('cuda'), add_noise=0):
    """
    Returns a tensor of special scatterings, associated with different wavelengths, and names of each scenario.
    Each item is a N-dimensional tensor, with N-1 entries being the scatterings, and another entry for the wavelength.
    """
    # special_scatterings = get_base_target_patterns(data_cfg, add_noise=add_noise)
    t, r, te, tm = utils.get_component_types_booleans(data_cfg)
    
    if te and tm:
        # special_scatterings = get_base_polarized_target_patterns(data_cfg, add_noise=add_noise)
        special_scatterings = get_polarized_target_patterns(data_cfg, add_noise=add_noise)
    else:
        special_scatterings = get_base_target_patterns(data_cfg, add_noise=add_noise)

    # scatterings = torch.cat([c for c in special_scatterings.values()], dim=0).to(device)
    scatterings = torch.stack([c for c in special_scatterings.values()], dim=0).to(device)
    lams = torch.tensor([float(l) for l in data_cfg.wavelengths], device=scatterings.device)
    scatterings_expanded = scatterings.repeat(len(lams), 1)
    # scatterings_expanded = scatterings_expanded[:, :data_cfg.info_t_orders**2] if data_cfg.use_t_only else scatterings_expanded[:, data_cfg.info_t_orders**2:data_cfg.info_t_orders**2+data_cfg.info_r_orders**2] if data_cfg.use_r_only else scatterings_expanded
    lams_expanded = lams.reshape(-1, 1).repeat(1, len(scatterings)).flatten()
    c0 = torch.cat([scatterings_expanded, lams_expanded.reshape(-1, 1)], dim=1)
    names = [
        f'{[c for c in special_scatterings.keys()][k % len(special_scatterings)]}_lam{lams_expanded[k].item():.3f}'
        for k in range(len(c0))]
    return c0, names


def main():
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np
    from data import data_config

    data = 'a3'

    data_cfg = data_config.get_data_cfg(data)
    diffractions = get_base_target_patterns(data_cfg)
    t_patterns = [(k, v.squeeze()[:data_cfg.info_t_orders**2].reshape(data_cfg.info_t_orders,data_cfg.info_t_orders).cpu()) for k,v in diffractions.items()]

    M = len(diffractions)

    # Create the figure
    s = 2
    f, axs = plt.subplots(2, M//2, figsize=(s*M//2, s*2))

    # Create a GridSpec with 2 rows and 8 columns to keep plots in the same size and allow centering
    # gs = gridspec.GridSpec(1, 8)
    vmin, vmax = 0.9*0.001, 0.4
    cmap = 'inferno'
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cbar_ticks = [0.0009, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4]
    cbar_ticklabels = ['< 0.001'] + [f'{t}' for t in cbar_ticks if t >= 0.001]

    # First row with 4 plots
    for i in range(M):
        row = i // (M//2)
        col = i % (M//2)
        T = t_patterns[i][1].clamp(min=0.00001)
        img = axs[row][col].imshow(T, norm=norm, cmap=cmap)
        ticks = [f'{tick}' for tick in np.arange(-(data_cfg.info_t_orders//2), (data_cfg.info_t_orders//2)+1)]
        axs[row][col].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=(row==1), left=True, right=False, labelleft=False)
        axs[row][col].tick_params(axis='y', which='both', bottom=True, top=False, labelbottom=False, left=True, right=False, labelleft=(col==0))
        axs[row][col].set_xticks(np.arange(data_cfg.info_t_orders), ticks)
        axs[row][col].set_yticks(np.arange(data_cfg.info_t_orders), reversed(ticks), rotation=90, va='center')
        # axs[row][col].set_title(f'$T_{{OOD}}^{{{i+1}}}$')
        if col == M//2 - 1:
            divider = make_axes_locatable(axs[row][col])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = f.colorbar(img, cax=cax)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)

    plt.tight_layout()
    plt.savefig(join(PROJECT_DIR, 'data', 'figs', f'{data}_target_patterns.png'), dpi=600)


if __name__ == '__main__':
    main()
