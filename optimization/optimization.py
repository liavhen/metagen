import os
import sys
sys.path.append('.')
sys.path.append('..')
# Import
import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import torcwa
from torchvision.utils import save_image
import evaluation.diffraction_measurement
from utils import utils
from evaluation.diffraction_measurement import *
import argparse
from utils.paths import *
from os.path import join
import pandas as pd
from data.data_config import *
from diffusion.sample import forward_model
from tqdm import tqdm
import json
from evaluation import metrics
from joblib import Parallel, delayed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")  # faster/safer headless plotting
torch.set_num_threads(1)



def save_curves_with_errors(df, nof_data_points, args):
    """ Notice: this function contains visualization bugs in the case of losses other that UE. """
    initial_mean = df.iloc[nof_data_points + 0, :]
    random_mean = df.iloc[nof_data_points + 1, :]
    initial_std = df.iloc[nof_data_points + 2, :]
    random_std = df.iloc[nof_data_points + 3, :]
    n = list(range(len(initial_mean)))
    nof_ticks = 10
    f = plt.figure(figsize=(6, 3))
    plt.locator_params(axis='x', nbins=nof_ticks)
    plt.plot(n, initial_mean, label=f'Initial Guess', color='#CC4F1B')
    plt.fill_between(n, initial_mean - initial_std, initial_mean + initial_std, alpha=0.3, facecolor='#FF9848')
    plt.plot(n, random_mean, label=f'Random Guess', color='#1B2ACC')
    plt.fill_between(n, random_mean - random_std, random_mean + random_std, alpha=0.3, facecolor='#089FFF')
    plt.title(f'Meta-surfaces Optimization with {args.lam}[$\\mu m$] Wavelength')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Uniformity Error')
    plt.ylim([0, 1.1])
    plt.xlim([0, args.num_steps])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(join(args.outdir, f'lam{args.lam}_optimization.png'), dpi=600)
    plt.close(f)


def plot_efficiencies(efficiencies ,label):
    """efficiencies is a list of k tuples, each contains two lists of length n: total and masked transmission"""
    efficiencies = tuple(zip(*efficiencies)) # convert from list of tuples to tuple of lists
    total_transmission, masked_transmission = efficiencies
    total_transmission = np.array(total_transmission).mean(axis=0)
    masked_transmission = np.array(masked_transmission).mean(axis=0)
    n = np.arange(len(total_transmission))
    nof_ticks = 10
    f = plt.figure(figsize=(6, 3))
    plt.locator_params(axis='x', nbins=nof_ticks)
    plt.plot(n, total_transmission, label=f'{label} Total Transmission', color='tab:blue')
    plt.plot(n, masked_transmission, label=f'{label} Masked Transmission', color='tab:purple')
    plt.title(f'Meta-surfaces Efficiencies Track')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Efficiency')
    plt.ylim([0, 1])
    plt.xlim([0, args.num_steps])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(join(args.outdir, f'efficiencies.png'), dpi=600)
    plt.close(f)


def optimize(target, h, lam, data_cfg, opt_cfg, outdir=None, initial_guess=None, verbose=True, optimize_h=False, **kwargs):

    tte, rte, ttm, rtm = utils.get_components_booleans(data_cfg)
    assert not (rte or rtm), "Reflections are not supported in the optimization procedure."
    dual_polarization = tte and ttm
    single_polarization = 'te' if tte and not ttm else 'tm' if ttm and not tte else None
    
    # Initialize x, h (optimization parameters)
    if initial_guess is None:
        x = torch.rand((data_cfg.resolution, data_cfg.resolution), dtype=torch.float32, device=torch.device('cuda'))
    else:
        alpha = 0.5 # noise amount coeff
        x = utils.threshold(torch.tensor(initial_guess, device=torch.device('cuda')), initial_guess.mean())
        noise = torch.rand_like(x)
        x = (1 - alpha) * x + alpha * noise
    
    x = x.detach().requires_grad_(True)
    h = torch.nn.Parameter(torch.tensor(h, device=x.device)).requires_grad_(optimize_h)

    # Initialize optimizer & scheduler
    optimizer = torch.optim.Adam([x] + optimize_h * [h], lr=opt_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_cfg.num_steps)
    
    # Perform the optimization.
    loss = torch.tensor(torch.inf, device=torch.device('cuda'))
    loss_history = []
    relative_error_history = []
    _range = tqdm(range(opt_cfg.num_steps), desc=f'Optimizing [loss: {opt_cfg.loss_type}]') if verbose else range(opt_cfg.num_steps)

    # Optimization loop.
    # ------------------
    beta_schedule = torch.exp(torch.linspace(0, 1, opt_cfg.num_steps, device=x.device) * np.log(1000.0))

    for i in _range:

        # Forward.
        optimizer.zero_grad(set_to_none=True)
        loss, (x_dict, actual, desired) = forward_model(
            x=x, h=h, lam=lam, data_cfg=data_cfg,
            desired_scattering=target, loss_type=opt_cfg.loss_type,
            beta=beta_schedule[i].item(), 
            min_feature_size=opt_cfg.min_feature_size, x_shrink=opt_cfg.x_shrink, y_shrink=opt_cfg.y_shrink,
        )
        
        loss_history.append(loss.item())
        relative_error_history.append(metrics.relative_error(utils.reconstruct_scatterings(actual, data_cfg)['Tte'], utils.reconstruct_scatterings(desired, data_cfg)['Tte']).item())
        
        # Log.
        if i % opt_cfg.log_every == 0:
            with torch.no_grad():
                _desired = utils.reconstruct_scatterings(desired, data_cfg)
                _actual = utils.reconstruct_scatterings(actual, data_cfg)
                if dual_polarization:
                    s1, s2, type =  _actual['Tte'].squeeze(0), _actual['Ttm'].squeeze(0), 'te-tm'
                else:
                    s1, s2, type =  _desired['T'+single_polarization].squeeze(0), _actual['T'+single_polarization].squeeze(0), 'desired-actual'
                
                utils.show_meta_atom_and_scatterings(x_dict['projected']*h.item(), s1, s2, type, data_cfg=data_cfg,
                                                    savepath=join(outdir, f'iter_{i}') if outdir is not None else None,
                                                    heights=data_cfg.heights, text_labels=True)
        
        # enforce h into the allowed heights range
        if optimize_h:
            h_loss = torch.clamp(h - max(data_cfg.heights), min=0) - torch.clamp(h - min(data_cfg.heights), max=0)
            loss = loss + h_loss
        
        # Backward.
        loss.backward()
        optimizer.step()
        scheduler.step()
    # ------------------

    # Save the final sample.
    if opt_cfg.log_every < np.inf:
        with torch.no_grad():
            loss, (x_dict, actual, desired) = forward_model(
                x=x, h=h, lam=lam, data_cfg=data_cfg,
                desired_scattering=target, loss_type=opt_cfg.loss_type,
                beta=10000, 
                min_feature_size=opt_cfg.min_feature_size, x_shrink=opt_cfg.x_shrink, y_shrink=opt_cfg.y_shrink,
            )
        _desired = utils.reconstruct_scatterings(desired, data_cfg)
        _actual = utils.reconstruct_scatterings(actual, data_cfg)
        if dual_polarization:
            s1, s2, type =  _actual['Tte'].squeeze(0), _actual['Ttm'].squeeze(0), 'te-tm'
        else:
            s1, s2, type =  _desired['T'+single_polarization].squeeze(0), _actual['T'+single_polarization].squeeze(0), 'desired-actual'
        
        utils.show_meta_atom_and_scatterings(x_dict['projected']*h.item(), s1, s2, type, data_cfg=data_cfg,
                                                savepath=join(outdir, f'iter_{i}') if outdir is not None else None,
                                                heights=data_cfg.heights, text_labels=True)


    # Log and save dictionary.
    sample = {}    
    h_min, h_max = min(data_cfg.heights), max(data_cfg.heights)
    scatterings = utils.reconstruct_scatterings(actual, data_cfg)
    desired = utils.reconstruct_scatterings(desired, data_cfg)
    sample['name'] = kwargs.get('target_name', f"optimized_sample") + f"_{utils.get_timestamp().replace('-', '')}_h{h.item():.3f}_lam{lam:.3f}"
    sample['layer'] = np.where(x_dict['projected'].detach().cpu().numpy() > 0.5, 1.0, -1.0)  # All layer are topologies only (-1, 1)
    sample['lvec'] = np.array([lam], dtype=np.float32)
    sample['h_original'] = np.array([h.detach().cpu()], dtype=np.float32)
    sample['h'] = np.array([utils.normalize_symmetric(h.detach().cpu(), h_max, h_min)], dtype=np.float32) # deprecated attributes
    sample['Tte'] = scatterings['Tte'].detach().cpu().numpy().astype(np.float32)
    sample['Rte'] = scatterings['Rte'].detach().cpu().numpy().astype(np.float32)
    sample['Ttm'] = scatterings['Ttm'].detach().cpu().numpy().astype(np.float32)
    sample['Rtm'] = scatterings['Rtm'].detach().cpu().numpy().astype(np.float32)
    sample['Tte_desired'] = desired['Tte'].detach().cpu().numpy().astype(np.float32)
    sample['Rte_desired'] = desired['Rte'].detach().cpu().numpy().astype(np.float32)
    sample['Ttm_desired'] = desired['Ttm'].detach().cpu().numpy().astype(np.float32)
    sample['Rtm_desired'] = desired['Rtm'].detach().cpu().numpy().astype(np.float32)
    sample['loss_type'] = opt_cfg.loss_type
    sample['loss_history'] = np.array(loss_history, dtype=np.float32)
    sample['relative_error_history'] = np.array(relative_error_history, dtype=np.float32)

    return sample


def random_guess_optimize_wrapper(k, c, target):
    outdir = join(c.exp_cfg.outdir, f'random_guess_{k}')
    os.makedirs(outdir, exist_ok=True)
    x_dict = optimize(target, c.main_params.h[k], c.main_params.lam, c.data_cfg, c.opt_cfg, outdir, optimize_h=True)
    torch.save(x_dict, join(outdir, f'{x_dict["name"]}.pt'))
    return x_dict

def initial_guess_optimize_wrapper(k, c, target, initial_guess, h):
    outdir = join(c.exp_cfg.outdir, f'initial_guess_{k}')
    os.makedirs(outdir, exist_ok=True)
    x_dict = optimize(target, h, c.main_params.lam, c.data_cfg, c.opt_cfg, outdir, initial_guess=initial_guess)
    torch.save(x_dict, join(outdir, f'{x_dict["name"]}.pt'))
    return x_dict

def main(c):
    tte, rte, ttm, rtm = utils.get_components_booleans(c.data_cfg)
    assert not (rte or rtm), "Reflections are not supported in the optimization procedure."
    
    if tte and ttm:
        # target = evaluation.get_polarized_target_patterns(c.data_cfg)[c.main_params.target].squeeze(0)
        target = evaluation.get_polarized_target_patterns(c.data_cfg)[c.main_params.target].squeeze(0)
    else:
        target = evaluation.get_base_target_patterns(c.data_cfg)[c.main_params.target].squeeze(0)

    # Collect initial guesses from a given dir (output of previous sampling procedure).
    initial_guesses_layers = None
    initial_guesses_heights = None
    if c.exp_cfg.initial_guess_pt is not None:
        pt = torch.load(c.exp_cfg.initial_guess_pt, weights_only=False)
        for k in pt.keys():
            batch_guesses_layers = pt[k]['sample'][-1, :, 0].cpu().numpy()
            batch_guesses_heights = pt[k]['sample'][-1, :, 1].mean(dim=(-2, -1)).cpu().numpy()
            initial_guesses_layers = batch_guesses_layers if initial_guesses_layers is None else np.concatenate((initial_guesses_layers, batch_guesses_layers), axis=0)
            initial_guesses_heights = batch_guesses_heights if initial_guesses_heights is None else np.concatenate((initial_guesses_heights, batch_guesses_heights), axis=0)
            num_guess_exps = min(len(initial_guesses_layers), c.exp_cfg.guess_exps)
    else:
        num_guess_exps = 0

    num_random_exps = c.exp_cfg.random_exps
    assert num_guess_exps > 0 or num_random_exps > 0, "No random or initial guesses provided. Please provide at least one of them."
    
    # Perform optimization for random guesses.
    all_losses_random_guess = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title("Loss vs. Optimization Steps")
    axes[1].set_title("T-TE Relative Error vs. Optimization Steps")
    if num_random_exps > 0:
        x_dicts = Parallel(n_jobs=min(c.exp_cfg.random_exps, 30))(
            delayed(random_guess_optimize_wrapper)(k, c, target) 
            for k in range(c.exp_cfg.random_exps))
        for k, x_dict in enumerate(x_dicts):
            axes[0].plot(x_dict['loss_history'], 'r', label=f"{x_dict['loss_type']} - random guess" if k == 0 else None)
            axes[1].plot(x_dict['relative_error_history'], 'r', label=f"{x_dict['loss_type']} - random guess" if k == 0 else None)
            all_losses_random_guess.append(x_dict['relative_error_history'])
        np.save(join(c.exp_cfg.outdir, 'all_losses_random_guess.npy'), np.array(all_losses_random_guess))
    else:
        all_losses_random_guess = np.ones((num_random_exps, c.opt_cfg.num_steps))

    # Perform optimization for initial guesses.
    all_losses_initial_guess = []
    
    if num_guess_exps > 0:
        x_dicts = Parallel(n_jobs=min(num_guess_exps, 30))(
            delayed(initial_guess_optimize_wrapper)(k, c, target, initial_guesses_layers[k], initial_guesses_heights[k]) 
            for k in range(num_guess_exps))
        for k, x_dict in enumerate(x_dicts):
            axes[0].plot(x_dict['loss_history'], 'b', label=f"{x_dict['loss_type']} - initial guess" if k == 0 else None)
            axes[1].plot(x_dict['relative_error_history'], 'b', label=f"{x_dict['loss_type']} - initial guess" if k == 0 else None)
            all_losses_initial_guess.append(x_dict['relative_error_history'])
        np.save(join(c.exp_cfg.outdir, 'all_losses_initial_guess.npy'), np.array(all_losses_initial_guess))
    else:
        all_losses_initial_guess = np.ones((num_random_exps, c.opt_cfg.num_steps))

    # Construct pandas dataframe with all the results
    all_ue = np.concatenate([np.array(all_losses_random_guess), np.array(all_losses_initial_guess)], axis=0)
    nof_random = len(all_losses_random_guess)
    nof_initial = len(all_losses_initial_guess)
    nof_total = nof_random + nof_initial
    df = pd.DataFrame(all_ue)
    df.loc[nof_total + 0, :] = df.iloc[nof_random:nof_total, :].mean(axis=0)
    df.loc[nof_total + 1, :] = df.iloc[:nof_random, :].mean(axis=0)
    df.loc[nof_total + 2, :] = df.iloc[nof_random:nof_total, :].std(axis=0)
    df.loc[nof_total + 3, :] = df.iloc[:nof_random, :].std(axis=0)

    df.to_csv(join(c.exp_cfg.outdir, f'all_{c.opt_cfg.loss_type}.csv'), index=True)

    args = EasyDict(lam=c.main_params.lam, num_steps=c.opt_cfg.num_steps, outdir=c.exp_cfg.outdir)
    save_curves_with_errors(df, nof_total, args)

    plt.legend()
    plt.tight_layout()
    plt.savefig(join(args.outdir, 'optimization_process.png'))
    plt.close(fig)


def process_args(args) -> EasyDict:
    c = EasyDict()
    
    # Build physical environment.
    c.data_cfg = get_data_cfg(args.data_cfg)
    # c.data_cfg = data_cfg
    if args.override:
        if args.p is not None:
            c.data_cfg.__dict__.update(periodicity=args.p)
        if args.r is not None:
            c.data_cfg.__dict__.update(resolution=args.r)
        if args.substrate is not None:
            c.data_cfg.__dict__.update(substrate=args.substrate)
        if args.structure is not None:
            c.data_cfg.__dict__.update(structure=args.structure)
    
    # Build optimization parameters.
    c.opt_cfg = EasyDict(
        loss_type=args.loss_type,
        lr=args.lr,
        min_feature_size=args.min_feature_size,
        x_shrink=args.x_shrink,
        y_shrink=args.y_shrink,
        num_steps=args.num_steps,
        log_every=args.log_every,
    )

    # Build experiment parameters.
    name = args.name+f'-{utils.get_timestamp()}' if args.name is not None else f'opt-{args.data_cfg}-{args.target}-{utils.get_timestamp()}'
    outdir = join(OPTIM_DIR, name)
    os.makedirs(outdir, exist_ok=True)

    c.exp_cfg = EasyDict(
        name=name,
        outdir=outdir,
        initial_guess_pt=args.initial_guess_pt,
        random_exps=args.random_exps,
        guess_exps=args.guess_exps if args.initial_guess_pt is not None else 0,
    )

        # Build main parameters.
    c.main_params = EasyDict(
        h=args.h if args.h is not None else np.random.uniform(min(c.data_cfg.heights), max(c.data_cfg.heights), size=c.exp_cfg.random_exps).tolist(),
        lam=args.lam,
        target=args.target,
    )

    with open(join(c.exp_cfg.outdir, 'config.json'), 'w') as f:
        dump = c.copy()
        dump.data_cfg = dump.data_cfg.__dict__
        json.dump(dump, f, indent=4)    
    
    return c

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="experiment name")

    # Main parameters.
    parser.add_argument("--h", type=float, default=None, help="height [um]")
    parser.add_argument("--lam", type=float, default=0.94, help="lambda (wavelegnth) [um]")
    parser.add_argument("--target", default='p_invariant_uniform', type=str,
                        help="key for the target scattering as encoded in "
                             "evaluation.quality_evaluation.get_polarized_target_patterns dictionary")
    
    # Physical environment.
    parser.add_argument("--data_cfg", type=str, default=None, help="complete data cfg")
    parser.add_argument("--override", action='store_true', default=False, help="If True, parameters in the data_cfg may be overridden by the following arguments")

    # Potential overrides.
    parser.add_argument("--substrate", type=str, default=None, help="material of the substrate", choices=['SiO2', 'Si', 'SiN', 'TiO2'])
    parser.add_argument("--structure", type=str, default=None, help="material of the structure", choices=['SiO2', 'Si', 'SiN', 'TiO2'])
    parser.add_argument("--r", type=int, default=None, help="resolution [pixels]")
    parser.add_argument("--p", type=float, default=None, help="periodicity [um]")
    parser.add_argument("--tet", type=float, default=0., help="theta (incident angle)")
    
    # Optimization parameters.
    parser.add_argument("--loss_type", type=str, default='mse', help="loss type for optimization")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial Learning Rate")
    parser.add_argument("--min_feature_size", type=float, default=0.5, help="Minimum feature size in [um]")
    parser.add_argument("--x_shrink", type=float, default=1, help="x axis shrink factor when smoothing (x > 1 encourages low frequencies on the x axis)")
    parser.add_argument("--y_shrink", type=float, default=1, help="y axis shrink factor when smoothing (y > 1 encourages low frequencies on the y axis)")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of optimization steps")
    parser.add_argument("--log_every", type=int, default=20, help="Logging frequency")
    
    # Experimental parameters.
    parser.add_argument("--initial_guess_pt", default=None, type=str, help="path to pt file for providing initial guess")
    parser.add_argument('--random_exps', type=int, default=30, help="number of random experiments")
    parser.add_argument('--guess_exps', type=int, default=30, help="max number of initial guess experiments")

    args = parser.parse_args()

    c = process_args(args)
    main(c)
