import torch
import torchvision
from torch import nn
from datetime import datetime
import os
from edm_utils.torch_utils import distributed as dist
# from distmap import euclidean_signed_transform
import numpy as np
from torchvision.transforms import Resize, InterpolationMode
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import *
from torch.fft import fft2, fftshift, ifft2, ifftshift
import warnings


# class SDF:
#     def __init__(self, sdf_decay=0.05):
#         self.sdf_decay = sdf_decay

#     def __call__(self, tensor):
#         assert isinstance(tensor, torch.Tensor), "Expecting input to be of type torch.Tensor!"
#         assert tensor.dim() == 4, "Expecting tensors of size [B, C, H, W]!"
#         assert tensor.min() == -1. and tensor.max() == 1., "Expecting tensors with values [-1.0, 1.0]"
#         return 2 * torch.sigmoid(self.sdf_decay * euclidean_signed_transform(tensor, ndim=2)) - 1


def apply_filter(x_fft, h_fft):
    return torch.real(ifft2(fftshift(x_fft * h_fft)))


def threshold(tensor, thresh=0.5):
    assert isinstance(tensor, torch.Tensor), "Expecting input to be of type torch.Tensor!"
    return torch.where(tensor >= thresh, 1., 0.)


def binary_projection(rho, beta, gamma=0.5):
    """ Pushes rho smoothly to values close to either 0 or 1. The smoothness depends on beta."""
    beta = torch.tensor(beta, dtype=rho.dtype, device=rho.device)
    gamma = torch.tensor(gamma, dtype=rho.dtype, device=rho.device)
    return (torch.tanh(beta*gamma) + torch.tanh(beta*(rho-gamma))) / (torch.tanh(beta*gamma) + torch.tanh(beta*(1-gamma)))


def smooth_rect_filter_fn(hbw, c=0, beta=3):
    """
    Returns a 1-D low/band pass filter (by design, should operate on r = sqrt(x^2 + y^2) with radial symmetry).
    c:      Central frequency. If c = 0, this is a low pass filter.
    hbw:    Half Band Width. If c - hbw < 0, this is a low pass filter, else this is a band-pass filter centered around c with half-band width `hbw` to each side.
    beta:   Smoothing factor. As beta increases, the filter approaces rectifying behaviour, whereas small value implies smooth decay.
    """
    return lambda x: (1/2 + 1/2 * torch.tanh(beta * (torch.abs(x + c) + hbw))) * (1/2 + 1/2 * torch.tanh(-beta * (torch.abs(x - c) - hbw)))


def gaussian_filter_fn(sigma):
    def gaussian_filter(x, sigma):
        g = torch.exp(-x**2/2/sigma**2)
        return g/g.max()
    return lambda x: gaussian_filter(x, sigma)


def ramp_filter_fn(R, kernel_size=21, dx=100, dy=100):
    """ 
    Ramp filter, averaging pixels by lineraly decaying weight of their neighboors.
    Addopted from 'Inverse design and demonstration of high-performance wide-angle diffractive optical elements', Kim et al. 2020
    Note: R, dx, dy are given in [nm].
    """
    def ramp_filter(x, R, kernel_size, dx, dy):
        x_grid = dx * torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=x.device)
        y_grid = dy * torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=x.device)
        vx, vy = torch.meshgrid(x_grid, y_grid, indexing='ij')
        r = torch.sqrt(vx**2 + vy**2)
        kernel = (R - r).clamp(min=0)
        kernel_fft = fftshift(fft2(kernel, s=x.shape))
        return kernel_fft.abs()/kernel_fft.abs().max()
    return lambda x: ramp_filter(x, R, kernel_size, dx, dy)


def sample_shifted_exponential(lam, start=1, n_samples=1, device=torch.device('cuda')):
    """ Generate samples from a shifted exponential distribution."""
    u = torch.rand(n_samples, device=device)  # Uniform samples in [0, 1)
    samples = start - torch.log(1 - u) / lam  # Inverse CDF transformation
    return samples


def sample_linearly_decreasing(a, b, n_samples=1, device=torch.device('cuda')):
    """ Returns a random number from the interval [a, b] with linearly decreasing probability."""
    u = torch.rand(n_samples, device=device)  # Uniform samples in [0, 1)
    return b - torch.sqrt((b - a) * (b - a) * u) # Inverse CDF transformation


def randn01(shape, device=torch.device('cuda')):
    """ Returns a tensor of shape `shape` with values with a approximate Gaussian distribution between 0 and 1. To avoid over-clamping, the std is fixed."""
    return torch.randn(shape, device=device).mul(0.25).clamp(-1,1).add(1).mul(0.5)
    

def rotate_image_on_meshgrid(image, f_x, f_y, theta):
    """
    Rotate an image computed on a meshgrid by an angle theta (in degrees).
    
    Args:
        image (torch.Tensor): 2D tensor representing the image.
        f_x (torch.Tensor): Meshgrid X-coordinates.
        f_y (torch.Tensor): Meshgrid Y-coordinates.
        theta (float): Rotation angle in degrees.
    
    Returns:
        torch.Tensor: Rotated image.
    """
    # Convert theta to radians
    theta_rad = torch.tensor((90+theta) * torch.pi / 180.0)

    # Define the rotation matrix
    rotation_matrix = torch.tensor([
        [torch.cos(theta_rad), -torch.sin(theta_rad)],
        [torch.sin(theta_rad), torch.cos(theta_rad)]
    ], device=image.device)
    
    # Stack original grid coordinates
    coords = torch.stack([f_x.flatten(), f_y.flatten()], dim=0)  # Shape: (2, N)

    # Apply the inverse rotation matrix to get back to original positions
    rotated_coords = torch.linalg.inv(rotation_matrix) @ coords  # Shape: (2, N)

    # Normalize rotated coordinates based on the original meshgrid range
    x_rot = rotated_coords[0, :].view_as(f_x)
    y_rot = rotated_coords[1, :].view_as(f_y)

    x_rot_norm = 2 * (x_rot - f_x.min()) / (f_x.max() - f_x.min()) - 1
    y_rot_norm = 2 * (y_rot - f_y.min()) / (f_y.max() - f_y.min()) - 1

    # Create a grid for grid_sample
    grid = torch.stack([x_rot_norm, y_rot_norm], dim=-1)  # Shape: (H, W, 2)

    # Unsqueeze image to have batch and channel dimensions for grid_sample
    image_unsqueezed = image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    # Use grid_sample to rotate the image
    rotated_image = torch.nn.functional.grid_sample(image_unsqueezed, grid.unsqueeze(0), align_corners=True, mode='bilinear', padding_mode='border')
    
    return rotated_image.squeeze()  # Remove batch and channel dimensions


def smoothing(rho, type, p, min_feature_size=0.5, **kwargs):
    """ 
    Wrapper for frequency-domain filtering for smoothing purposes. Supports only cyclic-symmetrical filtering.
    p is the size of the unit cell that is smoothed, given in [um].
    max_features_size determines the maximal characteristic size of possible features, and is given in [um].
    """
    
    rho_fft = fftshift(fft2(rho))
    
    x_shrink = kwargs.pop("x_shrink", 1)
    y_shrink = kwargs.pop("y_shrink", 1)
    theta =  kwargs.pop("theta", 0)

    assert rho.shape[-1] == rho.shape[-2], "The current implementation supports only square cells, but rho.shape[-1] != rho.shape[-2]"

    N = rho.shape[-1]
    d = p / N  # Pixel spacing in physical units
    freqs_x = fftshift(torch.fft.fftfreq(N, d=d, device=rho.device))  # Frequencies along x-axis
    freqs_y = fftshift(torch.fft.fftfreq(N, d=d, device=rho.device))  # Frequencies along y-axis
    f_x, f_y = torch.meshgrid(freqs_x, freqs_y, indexing="ij")

    smooth_distortion = torch.zeros_like(f_x)
    if kwargs.pop('add_filter_distortion', False):
        distrotion = max(x_shrink, y_shrink)/2 * torch.randn_like(f_x)
        symmetric_distortion = 0.5 * distrotion + 0.5 * distrotion.flip(0,1)
        smooth_distortion = smoothing(symmetric_distortion, type='smooth_rect', p=2/f_x.max())
    
    freq_magnitude = torch.sqrt((x_shrink*f_x)**2 + (y_shrink*f_y)**2) + smooth_distortion # Radial frequency magnitude
    freq_magnitude = rotate_image_on_meshgrid(freq_magnitude, f_x, f_y, theta)

    f_cutoff = 1 / min_feature_size

    if type == 'smooth_rect':
        if not kwargs.get('hbw', False):
            kwargs['hbw'] = f_cutoff
        h_fft = smooth_rect_filter_fn(**kwargs)(freq_magnitude)
        # return apply_filter(rho_fft, h_fft)
    
    elif type == 'gaussian':
        if not kwargs.get('sigma', False):
            kwargs['sigma'] = f_cutoff
        h_fft = gaussian_filter_fn(**kwargs)(freq_magnitude)
        # return apply_filter(rho_fft, h_fft)
    
    elif type == 'ramp':
        raise DeprecationWarning("Ramp filter is deprecated, use other filter instead.")
        if not kwargs.get('R', False):
            kwargs['R'] = f_cutoff      
        if not kwargs.get('dx', False):
            kwargs['dx'] = p/rho.shape[-1]    
        if not kwargs.get('dy', False):
            kwargs['dy'] = p/rho.shape[-2]   
        h_fft = ramp_filter_fn(**kwargs)(freq_magnitude)
    else:
        NotImplementedError
    
    return apply_filter(rho_fft, h_fft)

    

def print_dict_beautifully(d):
    string = ""
    for k, v in d.items():
        if isinstance(v, int):
            string += f"{k}: {v:6d}\t|\t"
        elif isinstance(v, float) or isinstance(v, np.float64):
            string += f"{k}: {v:.5f}\t|\t"
        else:
            string += f"{k}: {v}\t|\t"
    return string


def time():
    """ Returns the exact time of call (without date details)"""
    return datetime.now().time().replace(microsecond=0)


def now():
    """ Returns the exact time of call (including date details)"""
    return datetime.now().replace(microsecond=0)


def get_timestamp():
    """ Returns a parsered timestamp of the format: YYYYMMDD_HHMMSS"""
    return str(now()).replace('-', '').replace(':', '').replace(' ', '-')


def print0(logger, string):
    if dist.get_rank() == 0:
        logger.info(string)


def get_nof_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cuda_allocated_memory(device_idx=0):
    return round(torch.cuda.memory_allocated(device_idx)/1024**3,1), 'GB'

def get_component_types_booleans(data_cfg):
    """ Returns booleans (t, r, te, tm) indicating which component types are included in the data_cfg. """
    t = data_cfg.use_t_only or not data_cfg.use_r_only
    r = data_cfg.use_r_only or not data_cfg.use_t_only
    te = data_cfg.use_te_only or not data_cfg.use_tm_only
    tm = data_cfg.use_tm_only or not data_cfg.use_te_only
    return t, r, te, tm

def get_components_booleans(data_cfg):
    """ Returns booleans (tte, rte, ttm, rtm) indicating which specific components are included in the data_cfg. """
    t, r, te, tm = get_component_types_booleans(data_cfg)
    return t*te, r*te, t*tm, r*tm

def get_label_dim(data_cfg):
    Tte, Rte, Ttm, Rtm = get_components_booleans(data_cfg)
    num_t_components = Tte + Ttm
    num_r_components = Rte + Rtm
    return num_t_components * data_cfg.info_t_orders**2 + num_r_components * data_cfg.info_r_orders**2 + 1

def normalize01(x):
    return (x - x.min()) / (x.max() - x.min())


def normalize_symmetric(x, global_max=None, global_min=None):
    if global_max is None and global_min is None:
        global_max, global_min = x.max(), x.min()
    x = (x - global_min) / (global_max - global_min)  # scale to (0, 1)
    return 2*x - 1  # scale to (-1, 1)


def unnormalize_symmetric(x, global_max, global_min):
    a = 1/(global_max - global_min)
    b = global_min/(global_max - global_min)
    x = (x+1)/2
    return (x - b)/a


def save_images_batch(batch, step, path):
    y, n, y_plus_n, D_yn = batch
    stacked_image = torch.cat([
        viewable(y),
        viewable(n),
        viewable(y_plus_n),
        viewable(D_yn)
    ], dim=-1)
    torchvision.utils.save_image(
        tensor=stacked_image,
        fp=os.path.join(path, f"samples_from_t={step}.png"),
        padding=1,
        pad_value=1,
        nrow=4,
    )


def viewable(layer, sampled_from_model=False):
    """ Expects a layer to have at least 1 channel for performing element-wise multiplication """
    if sampled_from_model:
        return layer.mean(dim=(2,3))[:, 1].reshape(-1,1,1,1) * threshold(layer[:, 0, ...].unsqueeze(1), thresh=0.2)
    else:
        return layer.prod(dim=1, keepdim=True)


def state_dict_from_parallel_to_single(checkpoint, key='ema'):
    """
    In the case when the model was trained used DataParallel, and then reloaded without DataParallel wrapper,
    the state dict must be adjusted for compatibility, and that is what this function does.
    """
    from collections import OrderedDict
    model_state_dict = checkpoint[key]
    model_new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        model_new_state_dict[k.replace("module.", "")] = v
    checkpoint[key] = model_new_state_dict
    return checkpoint

# def get_data_cfg_scattering_mask(scattering, data_cfg):
#     """    
#     Expects `scattering` to be a Tensor with 2 dimensions (batch, scattering_dim). 
#     Masks entries based on data_cfg. The expected ordering is:
#     [Tte, Rte, Ttm, Rtm] - each of length 19*19=361.
#     It is possible and supported that there will be another entry corresponding with wavelegnth.

#     apply_augments: If 0, no augmentations are applied. If 1, TE is masked, if 2, TM is masked.
#     """
#     B = scattering.size(0)
#     wavelength_included = scattering.size(-1) == 4*19*19+1
#     # _scattering = scattering[:, :4*19*19].view(-1, 4, 19, 19)
#     mask = torch.ones((B, 4, 1, 1), device=scattering.device, dtype=torch.int32)
#     if data_cfg.use_r_only: # --> [0, Rte, 0, Rtm]
#         mask[:, ::2] = 0
#         # scattering[..., :l] = 0
#         # scattering[..., 2*l:3*l] = 0
    
#     elif data_cfg.use_t_only: # --> [Tte, 0, Ttm, 0]
#         mask[:, 1::2] = 0
#         # scattering[..., l:2*l] = 0 
#         # scattering[..., 3*l:4*l] = 0
    
#     if data_cfg.use_te_only:# or apply_augments == 2: # --> [Tte, Rte, 0, 0]
#         mask[:, 2:] = 0
#         # scattering[..., 2*l:4*l] = 0
    
#     if data_cfg.use_tm_only:# or apply_augments == 1: # --> [0, 0, Ttm, Rtm]
#         mask[:, :2] = 0
#         # scattering[..., :2*l] = 0

#     mask = mask.repeat(1, 1, 19, 19).view(B, -1)
#     if wavelength_included:
#         mask = torch.cat((mask, torch.ones(1)), dim=0)

#     return mask


def get_random_scattering_mask(data_cfg, max_masked=3, device=torch.device('cuda')):
    assert max_masked < sum(get_components_booleans(data_cfg)), "max_masked must be less than the number of scattering components to ensure that some information remains!"
    mask = torch.ones(4, device=device, dtype=torch.int32)
    a = torch.arange(4)
    p = torch.tensor([1,4,6,4])
    m = a[:max_masked+1][torch.multinomial(p[:max_masked+1].float(), 1)]
    mask[torch.randperm(4)[:m]] = 0  # Randomly select m indices to mask
    return mask



def match_masks(to_be_masked, already_masked, data_cfg):
    """ 
    Expects Tensors of shape [B, N] or [B, N-1] (N is the effective labels dimension, considering the data cfg, with or without 1 for wavelength).
    Masks `to_be_masked` based on `already_masked`.
    """
    label_dim = get_label_dim(data_cfg)
    assert to_be_masked.size(-1) in [label_dim, label_dim-1], f"Expecting to_be_masked to have shape [B, {label_dim}] or [B, {label_dim-1}], but got {to_be_masked.shape}!"
    assert already_masked.size() == to_be_masked.size(), f"Scattering tensors must have the same shape, but tensor 1 is {already_masked.shape} and tensor 2 is {to_be_masked.shape}!"

    n, m = data_cfg.info_t_orders, data_cfg.info_r_orders
    t, r, te, tm = get_component_types_booleans(data_cfg)

    # all components are included
    if to_be_masked.shape[-1] >= 2*n**2 + 2*m**2: 
        to_be_masked[:,                     : n**2              ] *= ~(already_masked[:,                       :               n**2] == 0).all(dim=-1).unsqueeze(-1)
        to_be_masked[:, n**2                : n**2 + m**2       ] *= ~(already_masked[:, n**2                  : n**2 + m**2       ] == 0).all(dim=-1).unsqueeze(-1)
        to_be_masked[:, n**2 + m**2         : n**2 + m**2 + n**2] *= ~(already_masked[:, n**2 + m**2           : n**2 + m**2 + n**2] == 0).all(dim=-1).unsqueeze(-1)
        to_be_masked[:, n**2 + m**2 + n**2  : -1                ] *= ~(already_masked[:, n**2 + m**2 + n**2    : -1                ] == 0).all(dim=-1).unsqueeze(-1)
    
    # Only TE,TM transmission are included
    elif to_be_masked.shape[-1] == 2*n**2 or to_be_masked.shape[-1] == 2*n**2 + 1:
        to_be_masked[:,      : n**2       ] *= ~(already_masked[:,      : n**2       ] == 0).all(dim=-1).unsqueeze(-1)
        to_be_masked[:, n**2 : n**2 + n**2] *= ~(already_masked[:, n**2 : n**2 + n**2] == 0).all(dim=-1).unsqueeze(-1)

    # Only TE,TM reflections are included
    elif to_be_masked.shape[-1] == 2*m**2 or to_be_masked.shape[-1] == 2*m**2 + 1:
        to_be_masked[:,      : m**2       ] *= ~(already_masked[:,      : m**2       ] == 0).all(dim=-1).unsqueeze(-1)
        to_be_masked[:, m**2 : m**2 + m**2] *= ~(already_masked[:, m**2 : m**2 + m**2] == 0).all(dim=-1).unsqueeze(-1)

    # Either TE or TM are included, with both transmission and reflection
    elif to_be_masked.shape[-1] == n**2 + m**2 or to_be_masked.shape[-1] == n**2 + m**2 + 1:
        to_be_masked[:,      : n**2       ] *= ~(already_masked[:,      : n**2       ] == 0).all(dim=-1).unsqueeze(-1)
        to_be_masked[:, n**2 : n**2 + m**2] *= ~(already_masked[:, n**2 : n**2 + m**2] == 0).all(dim=-1).unsqueeze(-1)

    # Else: a single component is included, thus it cannot be masked
    else:
        assert not (already_masked == 0).all(dim=-1).any(), f"Some pattern/s are masked, but only one T/RxTE/TM component is included (data_cfg={data_cfg.name})!"
    
    return to_be_masked

    # wavelength_included = to_be_masked.size(-1) == 4*19*19+1

    # _to_be_masked = to_be_masked[:, :4*19*19].clone().reshape(-1, 4, 19, 19)
    # _already_masked = already_masked[:, :4*19*19].clone().reshape(-1, 4, 19, 19)
    # _to_be_masked[(_already_masked == 0).all(dim=(2,3))] = 0
    # _to_be_masked = _to_be_masked.reshape(-1, 4*19*19)
    # if wavelength_included:
    #     _to_be_masked = torch.cat((_to_be_masked, to_be_masked[:, -1:]), dim=1)
    # return _to_be_masked


def eliminate_masked_patterns(scattering):
    """
    Expects a Tensor of shape [B, 19*19] or [B, 19,19].
    Returns a Tensor of shape [B', 19*19] or [B', 19, 19], where B' ≤ B denotes the non-zeroed patterns.
    """
    assert scattering.dim() in [2,3] and scattering.size(-1) == scattering.size(-2), "Expecting scattering to be of dim 2 or 3 with square shaped items."
    return scattering[(scattering!=0).any(dim=tuple(range(1, scattering.dim())))]


def pad_to_size(tensor, M=19):
    """ Expects a Tensor of shape [B, K, K] (for an arbitrary odd K) and returns a Tensor of shape [B, M, M]"""
    assert tensor.dim() == 3, "Expecting scattering to be of dim 3!"
    assert tensor.size(1) == tensor.size(2), "Expecting scattering to be square!"
    B, K, _ = tensor.size()
    assert K % 2 == 1, "Expecting scattering to have an odd size!"
    padding = (M - K) // 2
    return torch.nn.functional.pad(tensor, (padding, padding, padding, padding))


def eliminate_unsupported_components(scatterings, data_cfg):
    """ Expects a Tensor of shape [B, 4*19*19] (or [B, 4*19*19+1]) and returns a Tensor of shape [B, N] where N is the supported scattering dimension (AKA label_dim)"""
    # assert scatterings.dim() == 2, "Expecting scatterings to be of dim 2 (i.e batched)!"
    assert scatterings.size(-1) in [4*19*19, 4*19*19+1], f"Expecting scatterings to have shape [B, 4*19*19] or [B, 4*19*19+1], but got {scatterings.shape}!"
    B = scatterings.size(0) if scatterings.dim() > 1 else 1
    s_reshaped = scatterings.view(B, 4, 19, 19)
    include_indicators = list(get_components_booleans(data_cfg))
    orders = [data_cfg.info_t_orders, data_cfg.info_r_orders]
    return torch.cat([crop_around_center(s_reshaped[:, i], orders[i%2]).reshape(B, -1) for i in range(4) if include_indicators[i]], dim=-1)


def reconstruct_scatterings(scatterings, data_cfg):
    """ Expects a Tensor of shape [B, N] (or [B, N-1]) and returns a dictionary of T/R x TE/TM, where each key hold values of shape [B, 19, 19]"""    
    label_dim = get_label_dim(data_cfg)
    assert scatterings.dim() == 2, f"Expecting scatterings to be of dim 2 (i.e batched), but got shape {scatterings.shape}!"
    assert scatterings.size(-1) in [label_dim, label_dim-1], f"Expecting to_be_masked to have shape [B, {label_dim}] or [B, {label_dim-1}], but got {scatterings.shape}!"

    l = 19*19
    tte, rte, ttm, rtm = get_components_booleans(data_cfg)
    n, m = data_cfg.info_t_orders, data_cfg.info_r_orders

    output = {}

    # All components are included
    if all([tte, rte, ttm, rtm]):
        output['Tte'] = scatterings[..., :n**2].view(-1, n, n)
        output['Rte'] = scatterings[..., n**2 : n**2+m**2].view(-1, m, m)
        output['Ttm'] = scatterings[..., n**2+m**2 : n**2+m**2+n**2].view(-1, n, n)
        output['Rtm'] = scatterings[..., n**2+m**2+n**2 :n**2+m**2+n**2+m**2].view(-1, m, m)
     
    # TE only
    elif tte and rte and not (ttm or rtm):
        output['Tte'] = scatterings[..., :n**2].view(-1, n, n)
        output['Rte'] = scatterings[..., n**2:n**2+m**2].view(-1, m, m)
    
    # TM only
    elif ttm and rtm and not (tte or rte):
        output['Ttm'] = scatterings[..., :n**2].view(-1, n, n)
        output['Rtm'] = scatterings[..., n**2:n**2+m**2].view(-1, m, m)

    # T only
    elif tte and ttm and not (rte or rtm):
        output['Tte'] = scatterings[..., :n**2].view(-1, n, n)
        output['Ttm'] = scatterings[..., n**2:n**2+n**2].view(-1, n, n)
    
    # R only
    elif rte and rtm and not (tte or ttm):
        output['Rte'] = scatterings[..., :m**2].view(-1, m, m)
        output['Rtm'] = scatterings[..., m**2:m**2+m**2].view(-1, m, m)

    # A single component is included
    else: 
        if tte:
            output['Tte'] = scatterings[..., :n**2].view(-1, n, n)
        elif rte:
            output['Rte'] = scatterings[..., :m**2].view(-1, m, m)
        elif ttm:
            output['Ttm'] = scatterings[..., :n**2].view(-1, n, n)
        elif rtm:
            output['Rtm'] = scatterings[..., :m**2].view(-1, m, m)

    for k in ['Tte', 'Rte', 'Ttm', 'Rtm']:
        if k not in output:
            output[k] = torch.zeros((scatterings.size(0), 19, 19), device=scatterings.device)
        else:
            output[k] = pad_to_size(output[k], 19)

    return output


def scatterings_dict_to_vector(scatterings_dict, data_cfg):
    assert isinstance(scatterings_dict, dict), "Expecting scatterings_dict to be a dictionary!"
    for k in ['Tte', 'Rte', 'Ttm', 'Rtm']:
        assert k in scatterings_dict, f"Expecting scatterings_dict to contain {k} key!"
    
    # Extract the scattering components
    Tte = crop_around_center(scatterings_dict['Tte'], data_cfg.info_t_orders).reshape(-1, data_cfg.info_t_orders**2)
    Rte = crop_around_center(scatterings_dict['Rte'], data_cfg.info_r_orders).reshape(-1, data_cfg.info_r_orders**2)
    Ttm = crop_around_center(scatterings_dict['Ttm'], data_cfg.info_t_orders).reshape(-1, data_cfg.info_t_orders**2)
    Rtm = crop_around_center(scatterings_dict['Rtm'], data_cfg.info_r_orders).reshape(-1, data_cfg.info_r_orders**2)
    
    # Concatenate the components into a single tensor
    tte, rte, ttm, rtm = get_components_booleans(data_cfg)
    included_components = tte*[Tte] + rte*[Rte] + ttm*[Ttm] + rtm*[Rtm]
    scatterings = torch.cat(included_components, dim=-1)
    return scatterings


def flatten_conditions(T, R, resolution=19, r_nof_angles=5, t_nof_angles=3):
    """ Expects T,R to be Tensors of shape [res, res]"""
    mid_pix = resolution // 2
    t = T[mid_pix - t_nof_angles // 2: mid_pix + t_nof_angles // 2 + 1,
        mid_pix - t_nof_angles // 2: mid_pix + t_nof_angles // 2 + 1].reshape(-1, 1)
    r = R[mid_pix - r_nof_angles // 2: mid_pix + r_nof_angles // 2 + 1,
        mid_pix - r_nof_angles // 2: mid_pix + r_nof_angles // 2 + 1].reshape(-1, 1)
    return torch.cat((t, r), dim=0).reshape(1, -1)


def reshape_conditions(scattering, t=3, r=5):
    """
    Expects `scattering` to be a Tensor of shape (t**2+r**2+1,)
    corresponding to txt transmission orders, rxr reflection orders, and 1 wavelength
    """
    T = torch.zeros((19, 19), device=scattering.device)
    T[:t, :t] = scattering[:t**2].reshape(t, t)
    T = torch.roll(T, shifts=(19//2 - t//2, 19//2 - t//2), dims=(0, 1))

    R = torch.zeros((19, 19), device=scattering.device)
    try: # In case reflection is cut out of `scattering` due to data_cfg.use_t_only == True
        R[:r, :r] = scattering[t**2:-1].reshape(r, r)
        R = torch.roll(R, shifts=(19 // 2 - r // 2, 19 // 2 - r // 2), dims=(0, 1))
    except RuntimeError:
        pass
    return T, R


def align_cyclic_shift(cyc_shifted_img, aligned_img):
    """
    This functions aligns cyc_shifted_img with aligned_img.
    Assuming the two images are cyclic-shifted (and maybe noised) version of the same image.
    Expecting images of shape [B, C, H, W]
    """
    assert aligned_img.shape == cyc_shifted_img.shape, "Images must have the same shape!"
    # assert aligned_img.size(1) == 1, f"Expected 1 channel, got {aligned_img.size(1)}"
    fft1 = torch.fft.fft2(aligned_img)
    fft2 = torch.fft.fft2(cyc_shifted_img)
    cross_corr_argmax = torch.fft.ifft2(fft1*fft2.conj()).abs().flatten(start_dim=-2).argmax(dim=-1).detach()
    shifts = [divmod(idx[0].item(), cyc_shifted_img.shape[-1]) for idx in cross_corr_argmax]
    res = cyc_shifted_img.clone()
    for b in range(res.size(0)):
        res[b, 0] = torch.roll(cyc_shifted_img[b, 0], shifts[b], dims=(-2, -1))
    return res


def crop_around_center(img, crop_size_x, crop_size_y=None):
    crop_size_y = crop_size_x if crop_size_y is None else crop_size_y
    cx = img.shape[-1] // 2
    cy = img.shape[-2] // 2
    cropped = img[..., cy-crop_size_y//2:cy+crop_size_y//2+1, cx-crop_size_x//2:cx+crop_size_x//2+1]
    if (torch.abs(img.sum(dim=(-2, -1)) - cropped.sum(dim=(-2, -1))) > 1e-6).any():
        warnings.warn(f"Cropping around center with crop sizes ({crop_size_x}, {crop_size_y}) has cut out non-zero values (energy conservation failure) at indices: \
                      {(torch.abs(img.sum(dim=(-2, -1)) - cropped.sum(dim=(-2, -1))) > 1e-6).nonzero().flatten().cpu().numpy().tolist() if img.dim() == 3 else 0}")
    return cropped


def show_meta_atom_and_scatterings(meta_atom, s1, s2, type, data_cfg, savepath=None, text_labels=True, heights=None):

    # Initialize plots & coloring settings
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    vmin, vmax = 0.9*0.001, 0.5
    cmap = 'inferno'
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    heights = data_cfg.heights if heights is None else heights
    gray_cmap = plt.get_cmap('gray')

    # orders = s2.shape[0]

    # Plot
    ax0, ax1, ax2, ax3 = ax.ravel()
    ax0.set_title(f'Meta-atom\nh = {meta_atom.max().item():.3f} [um]')
    ax1.set_title(f'Metasurface (4x4)\np = {data_cfg.periodicity:.3f} [um]')
    if type == 't-r': # s1 = t, s2 = r
        ax2.set_title('Transmission Efficiencies\nat All Diffraction Orders')
        ax3.set_title('Reflection Efficiencies\nat All Diffraction Orders')
        s1_orders = data_cfg.info_t_orders
        s2_orders = data_cfg.info_r_orders
    elif type == 'desired-actual': # s1 = desired, s2 = actual
        ax2.set_title(f'Desired Transmission\n(total = {s1.sum().item():.3f})')
        ax3.set_title(f'Actual Transmission\n(total = {s2.sum().item():.3f})')
        s1_orders = data_cfg.info_t_orders
        s2_orders = data_cfg.info_t_orders
    elif type == 'te-tm': # s1 = Tte, s2 = Ttm
        ax2.set_title(f'TE Transmission\n(total = {s1.sum().item():.3f})')
        ax3.set_title(f'TM Transmission\n(total = {s2.sum().item():.3f})')
        s1_orders = data_cfg.info_t_orders
        s2_orders = data_cfg.info_t_orders
    elif type == 'Tte-Rtm': # s1 = Tte, s2 = Rtm
        ax2.set_title(f'TE Transmission\n(total = {s1.sum().item():.3f})')
        ax3.set_title(f'TM Reflection\n(total = {s2.sum().item():.3f})')
        s1_orders = data_cfg.info_t_orders
        s2_orders = data_cfg.info_r_orders
    
    s1 = crop_around_center(s1, s1_orders)
    s2 = crop_around_center(s2, s2_orders)

    cbar_ticks = [0.0009, 0.01, 0.05, 0.1, 0.2, 0.4, vmax]
    cbar_ticklabels = ['< 0.001'] + [f'{t}' for t in cbar_ticks[:-1] if t >= 0.001] + [f'>= {vmax}']

    # Visualize meta-atom
    meta_atom = meta_atom.squeeze().detach().cpu().numpy()
    img0 = ax0.imshow(meta_atom, cmap=gray_cmap, vmin=0, vmax=max(heights) if heights is not None else 1.45)
    ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    ax0.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = f.colorbar(img0, cax=cax)

    # Visualize metasurface
    metasurface = np.tile(meta_atom, (4, 4))
    img1 = ax1.imshow(metasurface, cmap=gray_cmap, vmin=0, vmax=max(heights) if heights is not None else 1.45)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    ax1.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = f.colorbar(img1, cax=cax)

    # Visualize s1
    s1_ticks = [f'{i}' for i in np.arange(-(s1_orders//2), (s1_orders//2)+1)]
    img2 = ax2.imshow(s1.clip(min=vmin).detach().cpu().numpy(), norm=norm, cmap=cmap)
    ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, left=False, right=False, labelleft=False)
    ax2.set_xticks(np.arange(s1_orders), s1_ticks)
    ax2.set_yticks(np.arange(s1_orders), reversed(s1_ticks), rotation=90, va='center')
    if text_labels:
        for k in range(s1.shape[0]):
            for j in range(s1.shape[1]):
                if s1[k, j] >= 0.01:
                    ax2.text(j, k, f'{s1[k, j]:.3f}'[1:], ha='center', va='center', color='white', fontsize=9)
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # f.colorbar(img2, cax=cax)


    # Visualize s2
    s2_ticks = [f'{i}' for i in np.arange(-(s2_orders//2), (s2_orders//2)+1)]
    img3 = ax3.imshow(s2.clip(min=vmin).detach().cpu().numpy(), norm=norm, cmap=cmap)
    ax3.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, left=False, right=False, labelleft=False)
    ax3.set_xticks(np.arange(s2_orders), s2_ticks)
    ax3.set_yticks(np.arange(s2_orders), reversed(s2_ticks), rotation=90, va='center')
    if text_labels:
        for k in range(s2.shape[0]):
            for j in range(s2.shape[1]):
                if s2[k, j] >= 0.01:
                    ax3.text(j, k, f'{s2[k, j]:.3f}'[1:], ha='center', va='center', color='white', fontsize=9)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = f.colorbar(img3, cax=cax)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)
    
    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600)
    plt.close(f)
    # print('Saved image.')


def binary_resize(x, size):
    return Resize((size[0], size[1]), interpolation=InterpolationMode.NEAREST_EXACT)(x)

