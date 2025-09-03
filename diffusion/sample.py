import argparse
import os
join = os.path.join
import numpy as np
import torch
from torchvision.utils import save_image
import random, copy
from tqdm import tqdm
from datetime import datetime
import logging, sys
sys.path.append('.')
sys.path.append('..')
from utils import utils
from utils.paths import *
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']
from edm_utils.dnnlib.util import EasyDict
# ----------------------------------------------------------------------------
# EDM sampler & EDM model
from torch.utils.data import DataLoader
from data.lmdb_dataset import MetaLensDatasetLMDB
from data.data_config import get_data_cfg
import diffusion
# ----------------------------------------------------------------------------
from evaluation.diffraction_measurement import torcwa_simulation
from evaluation.metrics import relative_error, uniformity_error, nrms_error
from evaluation.quality_evaluation import get_special_conditions
# ----------------------------------------------------------------------------
#from pnn.model import get_pnn
# ----------------------------------------------------------------------------
from schedulefree import AdamWScheduleFree
from diffusion.loss import EDMLoss
# ----------------------------------------------------------------------------
from ablations import cwgan, vae_lib


def print_(logger, string, verbose):
    if verbose:
        if logger is None:
            print(string)
        else:
            logger.info(string)


def rescale_noise_cfg(denoised_cfg, denoised_cond, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = denoised_cond.std(dim=list(range(1, denoised_cond.ndim)), keepdim=True)
    std_cfg = denoised_cfg.std(dim=list(range(1, denoised_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = denoised_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    denoised_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * denoised_cfg
    return denoised_cfg


def cfg_sample(net, x, t, labels, scale):
    uncond = net(x, t, None).to(torch.float32)
    if scale > 0.0:
        cond = net(x, t, labels).to(torch.float32)
        denoised = uncond + scale * (cond - uncond)
        denoised = rescale_noise_cfg(denoised, cond, guidance_rescale=scale)  # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        return denoised
    return uncond


def forward_model(x, h, lam, desired_scattering, data_cfg, loss_type='relative', **kwargs):
    """ 
    Performs a forward simulation of the given layer `x` with the given `desired_scattering`.
    x: torch.Tensor of shape (H, W)
    h: float, height of the layer [um]
    lam: float, wavelength of the light [um]
    desired_scattering: torch.Tensor of shape (N^2+1) where N is the number of transmission orders and 1 for wavelength
    """
    phy_kwargs = {
        'periodicity': data_cfg.periodicity,
        'h': h,
        'lam': lam,
        'tet': 0.0,
        'substrate': data_cfg.substrate,
        'structure': data_cfg.structure
    }
    
    smoothing_on = kwargs.get('smooth', True)
    
    smoothing_kwargs = dict(
        type='smooth_rect',
        p=phy_kwargs['periodicity'],
        min_feature_size=kwargs.get('min_feature_size', 0.5),
        x_shrink=kwargs.get('x_shrink', 1.0),
        y_shrink=kwargs.get('y_shrink', 1.0)
    )

    projecting_on = kwargs.get('project', True)

    projecting_kwargs = dict(
        beta=kwargs.get('beta', 10000),
        gamma=kwargs.get('gamma', 0.5)
    )

    # for _ in range(3):
    x_smoothed = utils.smoothing(x, **smoothing_kwargs) if smoothing_on else x
    x_projected = utils.binary_projection(x_smoothed, **projecting_kwargs) if projecting_on else x_smoothed
    x = x_projected
    
    scatterings = torcwa_simulation(phy_kwargs, layer=x, rcwa_orders=data_cfg.rcwa_orders, project=False) # project=False because either projecting_on=True or it is not desired
    
    if loss_type == 'relative':
        loss_fn = relative_error
    elif loss_type == 'ue':
        loss_fn = uniformity_error
    elif loss_type in ['ue+mse', 'mse+ue', 'mse-ue', 'ue-mse']:
        loss_fn_1 = uniformity_error
        loss_fn_2 = torch.nn.functional.mse_loss
        loss_fn = lambda actual, desired: loss_fn_1(actual, desired) + loss_fn_2(actual, desired)
    elif loss_type == 'nrms':
        loss_fn = nrms_error
    elif loss_type == 'mse':
        loss_fn = torch.nn.functional.mse_loss
    elif loss_type == 'l2norm':
        loss_fn = lambda actual, desired: torch.linalg.norm(actual - desired)
    elif loss_type == 'maximize':
        loss_fn = lambda actual, desired:  (-1*(desired > 0)*actual).sum()
    elif loss_type == 'maximize+mse':
        loss_fn_1 = lambda actual, desired:  (-1*(desired > 0)*actual).sum()
        loss_fn_2 = torch.nn.functional.mse_loss
        loss_fn = lambda actual, desired: loss_fn_1(actual, desired) + loss_fn_2(actual, desired)
    else:
        raise ValueError("Invalid loss function")
    
    # Cost Calculation
    # actual = actual * utils.get_data_cfg_scattering_mask(scatterings['S'], data_cfg)
    actual_scatterings = scatterings['all']
    actual_scatterings = utils.eliminate_unsupported_components(actual_scatterings, data_cfg)
    desired_scatterings = desired_scattering.unsqueeze(0)
    actual_scattering_masked = utils.match_masks(actual_scatterings, desired_scatterings, data_cfg=data_cfg)
    loss = loss_fn(actual_scattering_masked, desired_scatterings)

    x_dict = {
        'input': x,
        'smoothed': x_smoothed,
        'projected': x_projected
    }
    return loss, (x_dict, actual_scatterings, desired_scatterings)


def batch_forward_model(x0_hat, c, data_cfg, desc=''):    
    costs = torch.zeros(x0_hat.shape[0], device=x0_hat.device)
    pbar = tqdm(range(x0_hat.shape[0]), desc=desc)
    for b in pbar:
        xb = x0_hat[b,0]
        hb = x0_hat[b,1].mean() * max(data_cfg.heights)
        hb = hb.abs() if hb < 0 else hb
        lam, desired_scattering = c[b, -1], c[b, :-1]
        cost, _ = forward_model(xb, hb, lam, desired_scattering, data_cfg, loss_type='l2norm', smooth=False, project=False)
        costs[b] = cost
        pbar.set_postfix({'cost': cost.nanmean().item()}, refresh=False)
    return costs.nanmean()

from torchvision.transforms.functional import gaussian_blur
def blur_class_labels(class_labels_orig_dict, lams, blur_sigma, data_cfg):
    for k in class_labels_orig_dict.keys():
        class_labels_orig_dict[k] = gaussian_blur(class_labels_orig_dict[k], kernel_size=5, sigma=blur_sigma)
    vector = utils.scatterings_dict_to_vector(class_labels_orig_dict, data_cfg=data_cfg)
    blurred_class_labels = torch.cat([vector, lams.unsqueeze(-1)], dim=-1)
    return blurred_class_labels

@torch.no_grad()
def edm_sampler(
            net, latents, class_labels=None, randn_like=torch.randn_like,

            # Data Config (for RCWA Guidance)
            data_cfg=None,

            # General sampling Parameters
            num_steps=100, sigma_min=0.002, sigma_max=80, rho=7,
            S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,

            # Classifier-Free guidance
            cfg_scale=1.0,

            # Posterior Sampling
            posterior_sampling=False, lr_scale=1, num_posterior_steps=0,
    ):
        
        def denoise(x, t):
            cond = net(x, t, class_labels)
            if cfg_scale == 1:
                return cond, None
            uncond = net(x, t, None)
            interp = uncond.lerp(cond, cfg_scale)
            return interp, (cond, uncond)

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float32) * t_steps[0]
        X = [x_next]
        cost, h_cost = 0, 0
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next
            
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
                       
            if posterior_sampling and i >= num_steps - num_posterior_steps:
                with torch.enable_grad():
                    x_hat = x_hat.detach().requires_grad_(True)
                    denoised, _ = denoise(x_hat, t_hat)
                    try:

                        # measurement consistency loss
                        cost = batch_forward_model(x0_hat=denoised,  c=class_labels, data_cfg=data_cfg,  desc=f'Forward Model @ {i+1:3d}/{num_steps:3d}')

                        # h loss -  to prevent grads from pulling the heights outside the supported range
                        h = denoised[:, 1] * max(data_cfg.heights) # [h0, h1, h2 ...]
                        h_cost = (h - max(data_cfg.heights))[h > max(data_cfg.heights)].sum() + (min(data_cfg.heights) - h)[h < min(data_cfg.heights)].sum()

                        # total loss
                        cost = cost + h_cost
                        grads = torch.autograd.grad(outputs=cost, inputs=x_hat)[0]
                        grads = grads / torch.linalg.norm(grads, dim=(2,3), keepdim=True)

                    except RuntimeError as e:
                        print(f"Posterior sampling at step {i:3d}/{num_steps} failed with error: {e}")
                        grads = torch.zeros_like(x_hat)
                    
                    d_cur = ((x_hat - denoised) / t_hat)
                    x_next = x_hat + (t_next - t_hat) * d_cur 
                    x_next -= lr_scale * t_hat * grads
            else:
                # Euler step.
                denoised, _ = denoise(x_hat, t_hat)
                d_cur =  (x_hat - denoised) / t_hat 
                x_next = x_hat + (t_next - t_hat) * d_cur

            # MetaGen: we find the second order correction to be unnecessary in our case
            # # Apply 2nd order correction. 
            # if i < num_steps - 1:
            #     denoised, _ = denoise(x_next, t_next)
            #     d_prime = (x_next - denoised) / t_next
            #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            x_next = x_next.detach_()
            torch.cuda.empty_cache()

            X.append(denoised)
                
        return torch.stack(X)


def gan_sampler(G, latents, class_labels=None, **unexpected_kwargs):
    return torch.stack([G(latents, class_labels)]) # Using list to maintain compatibility with EDM samplers


def vae_sampler(vae, latents, class_labels=None, **unexpected_kwargs):
    return torch.stack([vae.sample(num_samples=latents.size(0), current_device=latents.device, labels=class_labels)])
    

def _get_model(data_cfg, **model_kwargs):
    if model_kwargs['model_type'] in ['SongUNet', 'DhariwalUNet', 'DiTL8', 'DiTB8', 'DiTS8']:
        return diffusion.get_model(data_cfg, **model_kwargs)
    elif model_kwargs['model_type'] in ["GAN", "DCGAN", "WGAN-CP", "WGAN-GP"]:
        return cwgan.get_model(data_cfg, **model_kwargs)
    elif model_kwargs['model_type'] in ["ConditionalVAE", "M-CVAE", "C-VQ-VAE"]:
        return vae_lib.get_model(data_cfg, **model_kwargs)
    else:
        raise NotImplementedError(f"Model type {model_kwargs['model_type']} is not implemented.")


def _get_dataloader(data_cfg, eval_batch_size=1, shuffle=True, size_limit=None, **unexpected_kwargs):
    dataset = MetaLensDatasetLMDB(data_cfg, size_limit=size_limit, augments=False)
    dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=shuffle, num_workers=10)
    return dataloader


def _get_ood_data(data_cfg, eval_batch_size=1, add_noise=0, specific_pattern=None, **unexpected_kwargs):
    """ Draws out-of-distribution samples ('target patterns')"""
    conditions, names = get_special_conditions(data_cfg=data_cfg, add_noise=add_noise)
    if specific_pattern is not None:
        conditions, names = torch.stack([c for i, c in enumerate(conditions) if specific_pattern in names[i]]), [name for name in names if specific_pattern in name]
        assert len(names) > 0, "A valid specific pattern must be provided!"
    if conditions.shape[0] == 1 and eval_batch_size > 1:
        conditions, names = conditions.repeat(eval_batch_size, 1), names * eval_batch_size
    layers = torch.zeros((conditions.shape[0], 1, 1, 1))
    return {'layer': layers, 'scattering': conditions, 'name': names}


def _get_data(data_cfg, dataloader=None, **data_kwargs):
    return dataloader.__iter__().__next__() if data_kwargs['data_type'] == 'test' else _get_ood_data(data_cfg, **data_kwargs)


def _get_latents(actual_batch_size=1, model_type='SongUNet', img_channels=2, img_resolution=64, latent_dim=100, **unexpected_kwargs):
    shape = [actual_batch_size]
    if model_type in ['SongUNet', 'DhariwalUNet', 'DiTL8', 'DiTB8', 'DiTS8']:
        shape.extend([img_channels, img_resolution, img_resolution])
    elif model_type in ["GAN", "DCGAN", "WGAN-CP", "WGAN-GP"]:
        shape.extend([latent_dim, 1, 1])
    return torch.randn(*shape)
    

@torch.no_grad()
def sample(
    data_cfg    = None, 
    model       = None,     
    model_kwargs    = {}, 
    dataloader  = None,     
    data_kwargs     = {}, 
    sampler_kwargs  = {},
    
    # Wrapper parameters
    repeat=1, reduction='mean', same_conditions=False, same_latents=False,

    # Misc parameters
    logger=None, verbose=False, **unexpected_kwargs

):

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    model = _get_model(data_cfg, **model_kwargs) if model is None else model
    if model_kwargs['model_type'] in ['SongUNet', 'DhariwalUNet', 'DiTL8', 'DiTB8', 'DiTS8', 'ConditionalVAE', 'M-CVAE', "C-VQ-VAE"]:
        model = model.eval().requires_grad_(False).to(device)
    elif model_kwargs['model_type'] in ["GAN", "DCGAN", "WGAN-CP", "WGAN-GP"]:
        model = model.G.eval().requires_grad_(False).to(device)

    if data_kwargs['data_type'] == 'test' and dataloader is None:
        dataloader = _get_dataloader(data_cfg, **data_kwargs)
        print_(logger, str(dataloader.dataset), verbose)

    drawn_conditions = False  # force same conditioning (scattering patterns) in each sampling
    generated_latent = False  # force using the same latent, maybe under different conditioning
    results = dict()

    for r in range(repeat):
        print_(logger, '--------------------------------------', verbose)
        print_(logger, f'Starts sampling #{r}...', verbose)
        
        if not (drawn_conditions and same_conditions):
            sample = _get_data(data_cfg, dataloader, **data_kwargs)
            layers, conditions, names = sample['layer'].to(device), sample['scattering'].to(device), sample['name']
            actual_batch_size = conditions.shape[0]
            drawn_conditions = True
        if not (generated_latent and same_latents):
            latents = _get_latents(actual_batch_size, **model_kwargs, **data_kwargs)
            latents = latents.to(device)
            if data_kwargs['data_type'] == 'ood':
                layers = torch.zeros((actual_batch_size, model_kwargs.get('img_channels', 2), model_kwargs.get('img_resolution', 64), model_kwargs.get('img_resolution', 64)), device=device)
            generated_latent = True
        
        print_(logger, f"Drawn samples: {names}", verbose)
        
        if model_kwargs['model_type'] in ['SongUNet', 'DhariwalUNet', 'DiTL8', 'DiTB8', 'DiTS8']:
            sampler_fn = edm_sampler
        elif model_kwargs['model_type'] in ["GAN", "DCGAN", "WGAN-CP", "WGAN-GP"]:
            sampler_fn = gan_sampler
        elif model_kwargs['model_type'] in ["ConditionalVAE", 'M-CVAE', "C-VQ-VAE"]:
            sampler_fn = vae_sampler
        else:
            raise NotImplementedError(f"Model type {model_kwargs['model_type']} is not implemented.")
                
        samples = sampler_fn(model, latents, conditions, data_cfg=data_cfg, **sampler_kwargs)

        possible_h = torch.tensor([float(h) for h in data_cfg.heights], device=device)
        special_conds_h = torch.tensor([possible_h[torch.randperm(len(possible_h))][0].item() for _ in range(latents.shape[0])], device=device)

        results[f'{r}'] = dict()
        results[f'{r}']['name'] = names
        results[f'{r}']['per'] = torch.tensor(data_cfg.periodicity, device=device).repeat(latents.shape[0])
        results[f'{r}']['h_original'] = sample['h_original'] if data_kwargs['data_type']=='test' else special_conds_h
        results[f'{r}']['h'] = sample['h'] if data_kwargs['data_type']=='test' else special_conds_h / max([float(h) for h in data_cfg.heights])
        results[f'{r}']['lvec'] = conditions[:, -1]
        results[f'{r}']['h_min'] = min([float(h) for h in data_cfg.heights])
        results[f'{r}']['h_max'] = max([float(h) for h in data_cfg.heights])
        results[f'{r}']['layer'] = layers
        results[f'{r}']['scattering'] = conditions
        results[f'{r}']['sample'] = samples

    return results


def evaluate_single_sample(result, b, data_cfg, h, lam):
    phy_kwargs = dict(periodicity=data_cfg.periodicity, h=h, lam=lam, tet=0.0, substrate=data_cfg.substrate, structure=data_cfg.structure)
    desired_scatterings = result['scattering'][b][...,:-1]
    layer = result['sample'][-1, b, 0].to(torch.float32).requires_grad_(False)
    with torch.no_grad():
        actual_scatterings = torcwa_simulation(phy_kwargs, layer=layer, rcwa_orders=data_cfg.rcwa_orders)
    return desired_scatterings, actual_scatterings['all']


def _stack(X, x):
    """ Stack x to X along dim 0. If X is empty (None), then x starts a new stack"""
    if isinstance(x, list):
        x = torch.stack(x, dim=0)
        return x if X is None else torch.cat([X, x], dim=0)
    return torch.stack([x], dim=0) if X is None else torch.cat([X, x.unsqueeze(0) if x.dim() < X.dim() else x], dim=0)

@torch.no_grad()
def compute_actual_scatterings(results, data_cfg, logger, n_jobs=8, verbose=True, to_cpu=False):

    for r in results.keys():
        desired_scatterings, actual_scatterings = None, None

        print_(logger, f'Evaluating batch of sampled results for {int(r)+1}/{len(results.keys())}', verbose)

        result = results[r]
        h_max = max(data_cfg.heights)
        h = h_max * result['sample'][-1, :, 1].mean(dim=(-2, -1)).cpu().numpy()
        lam = result['scattering'][:, -1].cpu().numpy()

        # avoiding numpy casting
        h = h.tolist()
        lam = lam.tolist()

        if n_jobs > 1:
            evals = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_single_sample)(result, b, data_cfg, h[b], lam[b])
                for b in range(len(result['name']))
            )

            desired_scatterings, actual_scatterings = map(list, zip(*evals))
            desired_scatterings = torch.stack(desired_scatterings, dim=0)         
            actual_scatterings = torch.stack(actual_scatterings, dim=0)                       

        else:
            for b in range(len(result['name'])):
                evals = evaluate_single_sample(result, b, data_cfg, h[b], lam[b])
                desired_scatterings = _stack(desired_scatterings, evals[0])
                actual_scatterings = _stack(actual_scatterings, evals[1])

        results[r]['desired_scatterings'] = desired_scatterings.cpu() if to_cpu else desired_scatterings
        results[r]['actual_scatterings'] = actual_scatterings.cpu() if to_cpu else actual_scatterings

    return results


apply_on_both = lambda fn,x,y: (fn(x), fn(y))

def compute_metrics(results, data_cfg):

    actual_scatterings = torch.cat([results[r]['actual_scatterings'] for r in results.keys()], dim=0)
    desired_scatterings = torch.cat([results[r]['desired_scatterings'] for r in results.keys()], dim=0)

    # Take off unsupported components from the actual scatterings
    actual_scatterings = utils.eliminate_unsupported_components(actual_scatterings, data_cfg)

    # Match masks of actual to desired
    actual_scatterings = utils.match_masks(actual_scatterings, desired_scatterings, data_cfg)

    # Re-order to separate Tte, Rte, Ttr, Rtr for different channels
    actual_dict = utils.reconstruct_scatterings(actual_scatterings, data_cfg) 
    desired_dict = utils.reconstruct_scatterings(desired_scatterings, data_cfg) 

    # Eliminate masked compontents
    Tte_actual, Tte_desired = apply_on_both(utils.eliminate_masked_patterns, actual_dict['Tte'], desired_dict['Tte'])
    Rte_actual, Rte_desired = apply_on_both(utils.eliminate_masked_patterns, actual_dict['Rte'], desired_dict['Rte'])
    Ttm_actual, Ttm_desired = apply_on_both(utils.eliminate_masked_patterns, actual_dict['Ttm'], desired_dict['Ttm'])
    Rtm_actual, Rtm_desired = apply_on_both(utils.eliminate_masked_patterns, actual_dict['Rtm'], desired_dict['Rtm'])

    # Compute relative errors.
    Tte_relative_errors = relative_error(Tte_actual, Tte_desired)
    Rte_relative_errors = relative_error(Rte_actual, Rte_desired)
    Ttm_relative_errors = relative_error(Ttm_actual, Ttm_desired)
    Rtm_relative_errors = relative_error(Rtm_actual, Rtm_desired)

    # Compute UE errors.
    Tte_ue_errors = uniformity_error(Tte_actual, Tte_desired)
    Rte_ue_errors = uniformity_error(Rte_actual, Rte_desired)
    Ttm_ue_errors = uniformity_error(Ttm_actual, Ttm_desired)
    Rtm_ue_errors = uniformity_error(Rtm_actual, Rtm_desired)

    # Pack everything into a dictionary.
    metrics = { 
        'Tte_relative_errors': Tte_relative_errors,
        'Rte_relative_errors': Rte_relative_errors,
        'Ttm_relative_errors': Ttm_relative_errors,
        'Rtm_relative_errors': Rtm_relative_errors,
        'T_mean_relative_error': torch.cat([Tte_relative_errors, Ttm_relative_errors], dim=0).nanmean().item(),
        'R_mean_relative_error': torch.cat([Rte_relative_errors, Rtm_relative_errors], dim=0).nanmean().item(),
        'TE_mean_relative_error': torch.cat([Tte_relative_errors, Rte_relative_errors], dim=0).nanmean().item(),
        'TM_mean_relative_error': torch.cat([Ttm_relative_errors, Rtm_relative_errors], dim=0).nanmean().item(),
        'mean_relative_error': torch.cat([Tte_relative_errors, Rte_relative_errors, Ttm_relative_errors, Rtm_relative_errors], dim=0).mean().item(),
        'Tte_ue_errors': Tte_ue_errors,
        'Rte_ue_errors': Rte_ue_errors,
        'Ttm_ue_errors': Ttm_ue_errors,
        'Rtm_ue_errors': Rtm_ue_errors,
        'T_mean_ue_error': torch.cat([Tte_ue_errors, Ttm_ue_errors], dim=0).nanmean().item(),
        'R_mean_ue_error': torch.cat([Rte_ue_errors, Rtm_ue_errors], dim=0).nanmean().item(),
        'TE_mean_ue_error': torch.cat([Tte_ue_errors, Rte_ue_errors], dim=0).nanmean().item(),
        'TM_mean_ue_error': torch.cat([Ttm_ue_errors, Rtm_ue_errors], dim=0).nanmean().item(),
        'mean_ue_error': torch.cat([Tte_ue_errors, Rte_ue_errors, Ttm_ue_errors, Rtm_ue_errors], dim=0).nanmean().item(),

    }

    return metrics


def log_results(results, outdir):

    torch.save(results, join(outdir, f'raw_data.pt'))
    
    for r in results.keys():

        original_layers = utils.viewable(results[r]['layer'])
        sampled_layers = utils.viewable(results[r]['sample'][-1], sampled_from_model=True)

        # Original layer vs. the generated sample
        cat_img = torch.cat([original_layers, sampled_layers], dim=-2)
        save_image(cat_img, f'{outdir}/image_{int(r) + 1}.png', pad_value=1, padding=1, nrow=len(original_layers), normalize=True)

        # Diffusion backward process
        process_img = results[r]['sample'].permute(1, 0, 2, 3, 4)  # get [B, T, C, H, W]
        B, T, C, H, W = process_img.size()
        process_img = torch.cat([utils.normalize01(process_img[:, t]) for t in range(T)], dim=-1)
        save_image(process_img, f'{outdir}/process_image_{int(r) + 1}.png', pad_value=1, padding=1, nrow=1, normalize=True)

        # Generated sample with periodicity of 4 in each direction
        repeat_orig_layer = original_layers.tile(1, 1, 4, 4)
        repeated_gen_sample = sampled_layers.tile(1, 1, 4, 4)
        repeated_img = torch.cat([repeat_orig_layer, repeated_gen_sample], dim=0)
        save_image(repeated_img, f'{outdir}/cell_image_{int(r) + 1}.png', padding=4, pad_value=0.5, nrow=len(sampled_layers), normalize=True)


def plot_results(results_lst, data_cfg, metric, outdir, text_labels=True):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm, PowerNorm
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    vmin, vmax = 0.9*0.001, 0.5
    cmap = 'inferno'
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

    # Configs    
    mid = 19 // 2
    t_orders = data_cfg.info_t_orders
    t_fontsize = 7 if data_cfg.info_t_orders == 11 else 9 if data_cfg.info_t_orders == 7 else 11
    t_ticks = [f'{i}' for i in np.arange(-(t_orders//2), (t_orders//2)+1)]
    cbar_ticks = [0.0009, 0.01, 0.05, 0.1, 0.2, 0.4, vmax]
    cbar_ticklabels = ['< 0.001'] + [f'{t}' for t in cbar_ticks[:-1] if t >= 0.001] + [f'>= {vmax}']
    booleans_names = ['Tte', 'Rte', 'Ttm', 'Rtm']
    booleans = utils.get_components_booleans(data_cfg)
    included_components = [b for i,b in enumerate(booleans_names) if booleans[i]]

    metric_fn = relative_error if 'relative' in metric else uniformity_error if 'ue' in metric else torch.nn.functional.mse_loss
    metric_name = 'Relative' if 'relative' in metric else 'Uniformity' if 'ue' in metric else 'L2'

    for idx, results in tqdm(list(enumerate(results_lst)), desc=f'[{metric_name}] Plotting results'):
        
        # Extract data.
        error, layer, sample, desired_scatterings, actual_scatterings, lam = results    
        desired_scatterings_dict = utils.reconstruct_scatterings(desired_scatterings, data_cfg)
        actual_scatterings_dict = utils.reconstruct_scatterings(utils.eliminate_unsupported_components(actual_scatterings, data_cfg), data_cfg)

        # Configure plot structure.
        ood = layer.max() == 0 and layer.min() == 0
        plots_widths = []
        plots_widths += (not ood) * [1.05]              # if in-distribution sampling - the original structure is available
        plots_widths += len(included_components) * [1]  # add room for each included component (desired scattering)
        plots_widths += [1, 1]                          # add room for the generate meta-atom and its corresponding 4x4 metasurface
        plots_widths += len(included_components) * [1]  # add room for each included component (actual scattering)
        plots_widths[-1] = 1.05                         # add room for the colorbar        
        
        fig = plt.figure(figsize=(3*len(plots_widths), 3.1))
        gs = gridspec.GridSpec(1, len(plots_widths), width_ratios=plots_widths)

        cursor = 0

        # Original layer (in case of in-distribution sampling)
        if not ood:
            ax = plt.subplot(gs[cursor])
            meta_atom = layer.squeeze(0,1).cpu().numpy() * max(data_cfg.heights)
            meta_surface = np.tile(meta_atom, (4, 4))
            img = ax.imshow(meta_surface, cmap='gray', vmin=0, vmax=max(data_cfg.heights))
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            ax.set_title(f'Original (4x4) Metasurface\n$h={meta_atom.max().item():.2f}, p={data_cfg.periodicity:.2f}$ [$\\mu m$]')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("left", size="5%", pad=0.05)
            clb = fig.colorbar(img, cax=cax)
            clb.ax.yaxis.set_ticks_position('left')
            clb.ax.yaxis.set_label_position('left')
            cursor += 1
        
        # Desired scatterings
        for comonent in included_components:
            ax = plt.subplot(gs[cursor])
            data = utils.crop_around_center(desired_scatterings_dict[comonent].squeeze(0), data_cfg.info_t_orders).clamp(min=vmin).cpu()
            img = ax.imshow(data, norm=norm, cmap=cmap)
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, left=False, right=False, labelleft=False)
            ax.set_xticks(np.arange(t_orders), t_ticks)
            ax.set_yticks(np.arange(t_orders), reversed(t_ticks), rotation=90, va='center')
            ax.set_title(f'Desired {comonent.capitalize()}\n$\\lambda = {lam:.3f}$ [$\\mu m$]')
            if text_labels:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        if data[i, j] >= 0.01:
                            ax.text(j, i, f'{data[i, j]:.2f}'[1:], ha='center', va='center', color='white' if data[i, j] <= vmax-0.1 else 'black', fontsize=t_fontsize)
            cursor +=1
        
        # Meta-atom and metasurface
        ax = plt.subplot(gs[cursor])
        data = sample.squeeze(0,1).cpu().numpy() * max(data_cfg.heights)
        img = ax.imshow(data, cmap='gray', vmin=0, vmax=max(data_cfg.heights))
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        ax.set_title(f'Generated Meta-atom\n$h={data.max().item():.2f}, p={data_cfg.periodicity:.2f}$ [$\\mu m$]')
        cursor += 1

        ax = plt.subplot(gs[cursor])
        data = np.tile(data, (4, 4))
        img = ax.imshow(data, cmap='gray', vmin=0, vmax=max(data_cfg.heights))
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        ax.set_title(f'4x4 Composition of the Metasurface\n$h={data.max().item():.2f}, p={data_cfg.periodicity:.2f}$ [$\\mu m$]')
        cursor += 1

        # Actual scatterings
        for comonent in included_components:
            ax = plt.subplot(gs[cursor])
            data = utils.crop_around_center(actual_scatterings_dict[comonent].squeeze(0), data_cfg.info_t_orders).clamp(min=vmin).cpu()
            img = ax.imshow(data, norm=norm, cmap=cmap)
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, left=False, right=False, labelleft=False)
            ax.set_xticks(np.arange(t_orders), t_ticks)
            ax.set_yticks(np.arange(t_orders), reversed(t_ticks), rotation=90, va='center')
            ax.set_title(f"Actual {comonent.capitalize()} ($\\lambda = {lam:.3f}$ [$\\mu m$])\n{metric_name} Error: {metric_fn(actual_scatterings_dict[comonent], desired_scatterings_dict[comonent])[0]:.3f}")
            if text_labels:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        if data[i, j] >= 0.01:
                            ax.text(j, i, f'{data[i, j]:.2f}'[1:], ha='center', va='center', color='white' if data[i, j] <= vmax-0.1 else 'black', fontsize=t_fontsize)
            cursor +=1

        # Add scatterings colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(img, cax=cax)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)

       
        plt.tight_layout()
        plt.savefig(join(outdir, f"{'labeled' if text_labels else 'unlabeled'}_results_{metric}_{idx}.png"), dpi=600)
        plt.close(fig)


def sample_pipeline(c, logger):

    if c.misc_kwargs.raw_data_path is not None:        
        results = torch.load(c.misc_kwargs.raw_data_path, weights_only=False)
    else: 
        tick = utils.now()
        results = sample(**c, **c.wrapper_kwargs, **c.misc_kwargs, logger=logger)
        print_(logger, f'Sampling took {utils.now() - tick} time.', c.misc_kwargs.verbose)

        tick = utils.now()
        results = compute_actual_scatterings(results, c.data_cfg, logger)
        print_(logger, f'Evaluation took {utils.now() - tick} time.', c.misc_kwargs.verbose)

        log_results(results, c.misc_kwargs.outdir)
    
    tick = utils.now()
    metrics = compute_metrics(results, c.data_cfg)
    print_(logger, f'Metric computations took {utils.now() - tick} time.', c.misc_kwargs.verbose)
    tick = utils.now()

    h_history = []
    h_max = max([float(h) for h in c.data_cfg.heights])

    if c.misc_kwargs.verbose:
        B = len(results['0']['name'])
        for r in results.keys():
            h = h_max * results[r]['sample'][-1, :, 1].mean(dim=(-2, -1)).cpu().numpy()
            h_history.extend(h)
            lam = results[r]['scattering'][:, -1].cpu().numpy()

    reduced_metrics = dict()
    optional_metrics = ['Tte_relative_errors', 'Ttm_relative_errors', 'Tte_ue_errors', 'Ttm_ue_errors']

    h_hist, bins = np.histogram(np.array(h_history), bins=100)
    f = plt.figure()
    plt.plot(bins[:-1], h_hist)
    plt.title(f'Histogram of h')
    plt.savefig(join(c.misc_kwargs.outdir, 'h_hist.png'))
    plt.close(f)

    for metric in optional_metrics:
        
        if c.wrapper_kwargs.reduction == 'best':
            t_errors, t_indices = metrics[metric].reshape(len(results.keys()), -1).min(dim=0)
        elif c.wrapper_kwargs.reduction == 'mean':
            t_errors = metrics[metric].reshape(len(results.keys()), -1).mean(dim=0)
            t_indices = torch.abs(metrics[metric].reshape(len(results.keys()), -1) - t_errors).argmin(dim=0) # take the sample that is closets to the mean
        else:
            raise NotImplementedError
        
        reduced_metrics[metric] = t_errors

        if c.misc_kwargs.verbose: # Save images
            metric_name = metric[:-7].replace('_', '-') if metric.endswith('errors') else metric.replace('_', '-')
            results_lst = []
            for b in range(len(t_errors)):
                best_attempt = t_indices[b]
                best_layer = utils.viewable(results[str(best_attempt.item())]['layer'][b:b + 1])
                best_sample = utils.viewable(results[str(best_attempt.item())]['sample'][-1][b:b + 1], sampled_from_model=True)
                best_desired_s = results[str(best_attempt.item())]['desired_scatterings'][b:b+1]
                best_actual_s = results[str(best_attempt.item())]['actual_scatterings'][b:b+1]
                lam = results[str(best_attempt.item())]['scattering'][:, -1].cpu().numpy()[b]
                results_lst.append((t_errors[b], best_layer, best_sample, best_desired_s, best_actual_s, lam))
            plot_results(results_lst, c.data_cfg, metric_name, c.misc_kwargs.outdir, text_labels=False)
            plot_results(results_lst, c.data_cfg, metric_name, c.misc_kwargs.outdir, text_labels=True)   

    return results, reduced_metrics


def get_args():
    parser = argparse.ArgumentParser()

    #---------------------------------------------------------------
    # Misc parameters
    #---------------------------------------------------------------
    parser.add_argument("--name", type=str, default='sampling-exp', help="experiment name")
    parser.add_argument("--raw_data_path", type=str, default=None, help="Path to the raw data (sampled a-priori) for skipping sampling and evaluation")
    parser.add_argument('--seed', default=None, type=int, help='global seed')
    parser.add_argument("--device_id", type=int, default=0, help="cuda device id")
    parser.add_argument('--verbose', action='store_true', default=False)


    #---------------------------------------------------------------
    # Model parameters
    #---------------------------------------------------------------
    # Model type
    parser.add_argument("--model_type", default='SongUNet', type=str, choices=['SongUNet', 'WGAN-GP', 'ConditionalVAE', 'M-CVAE', "C-VQ-VAE", 'DhariwalUNet', 'DiTL8', 'DiTB8', 'DiTS8'])
    
    # Common args
    parser.add_argument("--ckpts_path", default=None, type=str, help='full path to checkpoints dir')
    
    # models / case 1 - Diffusion models
    parser.add_argument('--model_channels', default=128, type=int, help='model_channels')
    parser.add_argument('--num_blocks', default=4, type=int, help='number of unet block per resolution')
    
    # models / case 2 - GAN/CVAE models
    parser.add_argument("--latent_dim", type=int, default=100, help="latent vectors dimension")


    #---------------------------------------------------------------
    # Data parameters
    #---------------------------------------------------------------
    parser.add_argument("--data_cfg", type=str, required=True)
    parser.add_argument("--img_channels", type=int, default=2, help="number of input channels")
    parser.add_argument("--img_resolution", type=int, default=None, help="image size (H,W) - assuming square images. If None - takes value from data_cfg")


    # Wrapper args
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--reduction", default='mean', type=str, choices=['mean', 'best'])
    parser.add_argument('--same_conditions', action='store_true', default=False)
    parser.add_argument('--same_latents', action='store_true', default=False)

    # Common args
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument('--override_wavelengths', nargs='+', default=None)
    parser.add_argument('--override_heights', nargs='+', default=None)
    
    # Data type
    parser.add_argument('--ood', action='store_true', default=False, help='Out-of-distribution sampling')
    
    # data / case 1 - In-distribution sampling
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument("--size_limit", type=int, default=None)

    # data / case 2 - Out-of-distribution sampling
    parser.add_argument('--specific_pattern', type=str, default=None)
    parser.add_argument('--add_noise', default=0.0, type=float, help='Multiplicative factor for adding noise to special conditions')

    #---------------------------------------------------------------
    # Sampling parameters
    #---------------------------------------------------------------    
    parser.add_argument('--num_steps', default=100, type=int, help='number of diffusion steps')
    parser.add_argument('--sigma_min', default=0.002, type=float, help='sigma_min')
    parser.add_argument('--sigma_max', default=100.0, type=float, help='sigma_max')
    parser.add_argument('--rho', default=3., type=float, help='Schedule hyper-parameter')
    parser.add_argument('--S_churn', default=0, type=float, help='Stochasticity in sampling trajectory')
    parser.add_argument('--S_noise', default=1., type=float, help='Standard deviation of noise in sampling trajectory')
    parser.add_argument('--cfg_scale', type=float, default=1)

    # sampling / case 1 - RCWA Guidance (Posterior Sampling)
    parser.add_argument('--posterior_sampling', action='store_true', default=False)
    parser.add_argument('--lr_scale', type=float, default=1)
    parser.add_argument('--num_posterior_steps', type=int, default=None, help="Number of posterior steps, performed lastly i.e on small noise levels")
    # parser.add_argument('--q', type=float, default=1.7, help="Sharpness of the ascendence in RCWA guidance schedule")  
    
    # # sampling / case 2 - Optimizable Forward Guidance
    # parser.add_argument('--fg', action='store_true', default=False, help="Use FG (Forward Guided) EDM sampler")
    # parser.add_argument("--K", type=int, default=1, help="Number of optimization steps in FG sampler")
    # parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for the optimization steps in FG sampler")
    # parser.add_argument("--loss_type", type=str, default='uniformity', choices=['uniformity', 'mse', 'relative', 'nrms'])

    # # sampling / case 3 - Test-time Training
    # parser.add_argument('--ttt', action='store_true', default=False, help="Use TTT (Test-Time Training) EDM sampler")
    # parser.add_argument("--M", type=int, default=5, help="Number of test time iterations in TTT sampler")
    # parser.add_argument("--batch_size_ttt", type=int, default=100, help="Batch size for the TTT sampler")
    # parser.add_argument("--lr_opt", type=float, default=0.01, help="Learning rate for the optimization steps in TTT sampler")
    # parser.add_argument("--optimization_steps", type=int, default=300)
    # parser.add_argument("--training_steps", type=int, default=1500)
    # parser.add_argument("--lr_diff", type=float, default=1e-5, help="Learning rate for the fine-tune in TTT sampler")
    # parser.add_argument("--ema_beta", type=float, default=0.8, help="Decay rate for the EMA in TTT sampler")    
        
    args = parser.parse_args()

    return regroup_args(args)




def regroup_args(args) -> EasyDict:
    
    c = EasyDict()

    # Arrange misc args
    args.outdir = os.path.join(SAMPLINGS_DIR, f"{args.name}_{utils.get_timestamp()}")
    args.seed = torch.random.seed() if args.seed is None else args.seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    c.misc_kwargs = EasyDict(name=args.name, outdir=args.outdir, raw_data_path=args.raw_data_path, device_id=args.device_id, seed=args.seed, verbose=args.verbose)

    # Arrange wrapper args
    c.wrapper_kwargs = EasyDict(repeat=args.repeat, reduction=args.reduction, same_conditions=args.same_conditions, same_latents=args.same_latents)

    # Arrange data args
    data_cfg = get_data_cfg(args.data_cfg) if isinstance(args.data_cfg, str) else args.data_cfg
    data_cfg.heights = args.override_heights if args.override_heights is not None else data_cfg.heights
    data_cfg.wavelengths = args.override_wavelengths if args.override_wavelengths is not None else data_cfg.wavelengths
    c.update(data_cfg=data_cfg)
    c.data_kwargs = EasyDict(data_type='ood' if args.ood else 'test', eval_batch_size=args.eval_batch_size)
    if c.data_kwargs.data_type == 'test':
        c.data_kwargs.update(shuffle=args.shuffle,size_limit=args.size_limit)
    else:
        c.data_kwargs.update(specific_pattern=args.specific_pattern, add_noise=args.add_noise)
        assert not (
            c.data_kwargs.data_type == 'ood' and 
            c.data_kwargs.specific_pattern is None and 
            c.data_kwargs.eval_batch_size > 1
            ), "When sampling OOD without specific pattern, eval_batch_size must be 1."
        
    # Arrange model arguments
    label_dim = utils.get_label_dim(c.data_cfg)
    c.model_kwargs = EasyDict(model_type=args.model_type, label_dim=label_dim, img_channels=args.img_channels, img_resolution=args.img_resolution if args.img_resolution is not None else c.data_cfg.resolution)
    if args.ckpts_path is not None:
        ckpts_path = args.ckpts_path
    else:
        if c.model_kwargs.model_type == 'SongUNet':
            ckpts_path = c.data_cfg.ckpts_path.metagen
        elif c.model_kwargs.model_type == "WGAN-GP":
            ckpts_path = c.data_cfg.ckpts_path.cwgan
        elif c.model_kwargs.model_type == 'M-CVAE':
            ckpts_path = c.data_cfg.ckpts_path.cvae
        else:
            raise ValueError(f"Checkpoint must be explicitly provided if model_type = {c.model_kwargs.model_type}!")
    c.model_kwargs.update(ckpts_path=ckpts_path)
    if c.model_kwargs.model_type in ['SongUNet', 'DhariwalUNet', 'DiTL8', 'DiTB8', 'DiTS8']:
        c.model_kwargs.update(model_channels=args.model_channels, num_blocks=args.num_blocks)
    elif c.model_kwargs.model_type in ["GAN", "DCGAN", "WGAN-CP", "WGAN-GP", "ConditionalVAE", 'M-CVAE', "C-VQ-VAE"]:
        c.model_kwargs.update(latent_dim=args.latent_dim)
        
    # Arrange sampler arguments
    # args.S_churn = args.num_steps / 4 if args.S_churn is None else args.S_churn
    c.sampler_kwargs = EasyDict(num_steps=args.num_steps, rho=args.rho, S_churn=args.S_churn, S_noise=args.S_noise, cfg_scale=args.cfg_scale)
    if args.posterior_sampling:
        num_posterior_steps = args.num_posterior_steps if args.num_posterior_steps is not None else args.num_steps//2
        c.sampler_kwargs.update(posterior_sampling=True, lr_scale=args.lr_scale, num_posterior_steps=num_posterior_steps)
    
    return c


def main():
    
    # Free any unused GPU memory
    torch.cuda.empty_cache()
    
    c = get_args()

    os.makedirs(c.misc_kwargs.outdir, exist_ok=True)

    logging.basicConfig(
        filename=f'{c.misc_kwargs.outdir}/eval.log',
        filemode='a+',
        format='%(asctime)s %(levelname)s --> %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info("#################### Arguments: ####################")
    logger.info('\n'.join(f'{k}\n: {str(v)}' for k,v in c.items()))

    results, reduced_metrics = sample_pipeline(c, logger)

    logger.info(f"\nTotal Errors (K = {len(results.keys())}, reduction={c.wrapper_kwargs.reduction}):")
    logger.info(f"\t Tte Relative Error:\t{reduced_metrics['Tte_relative_errors'].nanmean().item():.4f} "
                f"\u00B1 {reduced_metrics['Tte_relative_errors'].std().item() if len(reduced_metrics['Tte_relative_errors']) > 1 else None}"
    )
    logger.info(f"\t Ttm Relative Error:\t{reduced_metrics['Ttm_relative_errors'].nanmean().item():.4f}"
                f" \u00B1 {reduced_metrics['Ttm_relative_errors'].std().item() if len(reduced_metrics['Ttm_relative_errors']) > 1 else None}"
    )
    logger.info(f"\t Tte UE Error:\t\t{reduced_metrics['Tte_ue_errors'].mean().item():.4f}"
                f" \u00B1 {reduced_metrics['Tte_ue_errors'].std().item() if len(reduced_metrics['Tte_ue_errors']) > 1 else None}"
    )
    logger.info(f"\t Ttm UE Error:\t\t{reduced_metrics['Ttm_ue_errors'].mean().item():.4f}"
                f" \u00B1 {reduced_metrics['Ttm_ue_errors'].std().item() if len(reduced_metrics['Ttm_ue_errors']) > 1 else None}"
    )

    logger.info(f"#################### Done! ####################")


if __name__ == "__main__":
    tick = utils.now()
    main()
    print(f'Script took {utils.now()-tick} time.')
