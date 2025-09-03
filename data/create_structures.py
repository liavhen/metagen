import os
import sys
sys.path.append('.')
sys.path.append('..')

# -----------------------------------------------------
from os.path import join
from joblib import Parallel, delayed
import argparse, shutil
from tqdm import tqdm
# -----------------------------------------------------
import numpy as np
import torch
import lmdb, pickle, zlib
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torchvision.transforms.functional import rotate, center_crop
import torch.nn.functional as F
# -----------------------------------------------------
from utils.paths import *
from evaluation.diffraction_measurement import torcwa_simulation
from optimization.optimization import optimize
from evaluation.quality_evaluation import *
from utils import utils
from data import data_config
import scipy.io as sio
from edm_utils.dnnlib.util import EasyDict
# -----------------------------------------------------

# Source 1: Pre-defined pseudo free-form dataset.
freeform_dir = join(DATA_DIR, 'freeform_patterns')
freeform_files = os.listdir(freeform_dir)
def generate_pseudo_freeform(idx):
    layer = np.load(join(freeform_dir, freeform_files[idx]))
    layer = torch.from_numpy(layer).float().cuda().unsqueeze(0).unsqueeze(0)
    
    # Maybe squeeze to half.
    if torch.rand(1) < 0.5:
        size = layer.shape[-1]
        if torch.rand(1) < 0.5:
            layer = utils.binary_resize(layer, (size, size//2))
            layer = F.pad(layer, (0, size//2), "constant", 0)
        else:
            layer = utils.binary_resize(layer, (size//2, size))
            layer = F.pad(layer, ( 0, 0,size//2, 0), "constant", 0)

    # Maybe rotate.    
    theta = float(90 * (torch.rand(1) < 0.5))
    layer = rotate(layer, theta).squeeze(0,1)

    return layer

# Source 2: Noise Filtering.
def generate_filtered_noise(image_size, p, device=torch.device('cuda')):
    # Generate random noise image.
    large_image_size = int(2**0.5 * image_size)
    noise = torch.rand(size=(1, 1, large_image_size, large_image_size), device=device)

    # Create a pseudo-symmetric noise pattern by convexly combining the original and its flipped version
    gamma = np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    noise_flipped = torch.flip(noise, dims=(-1,))
    noise = gamma * noise + (1-gamma) * noise_flipped
    phi_shift = 90 * (torch.rand(1) < 0.5)
    phi = float(45*torch.randn(1) + phi_shift)
    noise = rotate(noise, phi)
    noise = center_crop(noise, (image_size, image_size)).squeeze(0,1)

    # Smoothen the noise pattern in Fourier space
    min_feature_size = float(torch.rand(1) * (p/2 - max(p/10, 0.25)) + max(p/10, 0.25))
    s = utils.sample_shifted_exponential(n_samples=1, lam=1, device=noise.device)
    x_shrink, y_shrink = (s, 1) if torch.rand(1)<0.5 else (1, s)   
    theta = float(90*torch.rand(1))
    duty_cycle = float(torch.rand(1) * (0.6 - 0.4) + 0.4)

    # Iterative smoothing and projection
    for _ in range(5):
        filtered = utils.smoothing(noise, p=p, type='smooth_rect',min_feature_size=min_feature_size, x_shrink=x_shrink, y_shrink=y_shrink, theta=theta)
        noise =  1 * (filtered > (torch.quantile(filtered, duty_cycle)))
    
    return noise

# Source 3: Parameterized eliptic.
def generate_eliptic(image_size, output_periods=False, device=torch.device('cuda')):
    Py = np.random.choice([1,2], p=[0.8, 0.2])  
    Px = np.random.choice([1,2], p=[0.8, 0.2])  
    canvas = torch.zeros((image_size, image_size), device=device)
    for i in range(Py):
        for j in range(Px):
            c = image_size // 2
            r = utils.randn01(1, device=device) * (0.5*image_size - 0.05*image_size) + 0.05*image_size # radius
            sx = torch.rand(1, device=device) * (2-1) + 1 # squeeze factor
            sy = torch.rand(1, device=device) * (2-1) + 1 # squeeze factor
            # Initialize a blank canvas
            o = torch.arange(-c, c, device=device) + 1
            x,y = torch.meshgrid(o, o, indexing='ij')
            subcanvas = torch.zeros_like(canvas)
            subcanvas[(sx*x)**2 + (sy*y)**2 < r**2] = 1
            subcanvas = subcanvas[::Py, ::Px]
            canvas[int(i*image_size/Py) : int((i+1)*image_size/Py), int(j*image_size/Px) : int((j+1)*image_size/Px)] = subcanvas
    if output_periods:
        return canvas, Py, Px
    return canvas


def generate_rectangles(image_size, output_periods=False, device=torch.device('cuda')):
    Py = np.random.choice([1,2], p=[0.8, 0.2])  
    Px = np.random.choice([1,2], p=[0.8, 0.2])  
    canvas = torch.zeros((image_size, image_size), device=device)
    for i in range(Py):
        for j in range(Px):
            c = image_size // 2
            h = int(utils.randn01(1, device=device) * (0.95*image_size - 0.05*image_size) + 0.05*image_size) # height
            w = int(utils.randn01(1, device=device) * (0.95*image_size - 0.05*image_size) + 0.05*image_size) # width
            subcanvas = torch.zeros_like(canvas)
            subcanvas[c - h//2 : c + h//2, c - w//2 : c + w//2] = 1
            subcanvas = subcanvas[::Py, ::Px]
            canvas[int(i*image_size/Py) : int((i+1)*image_size/Py), int(j*image_size/Px) : int((j+1)*image_size/Px)] = subcanvas
    if output_periods:
        return canvas, Py, Px
    return canvas


def generate_jerusalem_cross(image_size, output_periods=False, device=torch.device('cuda')):
    Py = np.random.choice([1,2], p=[0.8, 0.2])  
    Px = np.random.choice([1,2], p=[0.8, 0.2])  
    canvas = torch.zeros((image_size, image_size), device=device)
    for i in range(Py):
        for j in range(Px):
            c = image_size // 2
            l = int(torch.rand(1) * (image_size - image_size/4) + image_size/4) # length of the cross arms
            w = int(torch.rand(1) * (image_size/4 - image_size/8) + image_size/8) # width of the cross arms
            b = int(torch.rand(1) * (image_size/2 - image_size/4) + image_size/4) # handles length
            bw = int(torch.rand(1) * (image_size/8 - image_size/16) + image_size/16) # handles width

            # Initialize a blank canvas
            subcanvas = torch.zeros_like(canvas)

            # Draw the main cross
            subcanvas[c - l//2 : c + l//2, c - w//2 : c + w//2] = 1
            subcanvas[c - w//2 : c + w//2, c - l//2 : c + l//2] = 1
            
            # Add bars at the end of each arm
            subcanvas[c - b//2 : c + b//2 , c + l//2 - bw : c + l//2] = 1  # Right bar
            subcanvas[c - b//2 : c + b//2 , c - l//2 :c - l//2 + bw] = 1  # Left bar
            subcanvas[c + l//2 - bw : c + l//2 , c - b//2 : c + b//2] = 1  # Bottom bar
            subcanvas[c - l//2 : c - l//2 + bw , c - b//2 : c + b//2] = 1  # Bottom bar

            # subcanvas = subcanvas[::Py, ::Px]
            subcanvas = torch.nn.functional.interpolate(
                subcanvas.unsqueeze(0).unsqueeze(0), 
                size=(image_size//Py, image_size//Px), 
                mode='nearest-exact'
            ).squeeze(0,1)

            canvas[int(i*image_size/Py) : int((i+1)*image_size/Py), int(j*image_size/Px) : int((j+1)*image_size/Px)] = subcanvas

    if output_periods:
        return canvas, Py, Px
    return canvas


def generate_artificial_periods(canvas):
    Py = np.random.choice([1,2], p=[0.8, 0.2])  
    Px = np.random.choice([1,2], p=[0.8, 0.2])
    canvas = canvas[::Py, ::Px]
    return torch.tile(canvas, (Py, Px))


def generate_fn_wrapper(generate_fn, **kwargs):
    if generate_fn == generate_filtered_noise:
        kwargs.pop('output_periods', None)
        layer = generate_fn(**kwargs)
    elif generate_fn == generate_pseudo_freeform:
        idx = np.random.randint(0, 100000)
        layer = generate_fn(idx)
    elif generate_fn == generate_grating_profile:
        layer = generate_fn(**kwargs)
        layer = generate_artificial_periods(layer)
    else:
        layer, Py, Px = generate_fn(kwargs.pop('image_size', 64), output_periods=True, device=kwargs.pop('device', torch.device('cuda')))
        if Py == 1 and Px == 1:
            layer = generate_artificial_periods(layer)
    return layer


@torch.no_grad()
def get_data_sample(idx, data_cfg, lam_str, device=torch.device('cuda')):
    
    # Choose generation source randomly.
    generate_fn = np.random.choice([
        generate_filtered_noise, 
        generate_eliptic, 
        generate_rectangles, 
        generate_jerusalem_cross,
        generate_pseudo_freeform,
        generate_grating_profile], 
        p=[0.8, 0.03, 0.03, 0.00, 0.07, 0.07])
        # p=[0.0, 0.0, 0.0, 0.0, 1.0])
    
    # Pre-configure.
    sample = {}
    lam = float(lam_str)
    
    # Generate layer.
    kwargs = dict(image_size=data_cfg.resolution, p=data_cfg.periodicity, device=device)
    layer = generate_fn_wrapper(generate_fn, **kwargs)
    while len(layer.unique()) <= 1:
        layer = generate_fn_wrapper(generate_fn, **kwargs)    
    
    # Simulate.
    h_min, h_max = min(data_cfg.heights), max(data_cfg.heights)
    h = np.random.uniform(h_min, h_max)
    phy_kwargs = dict(periodicity=data_cfg.periodicity, h=h, lam=lam, tet=0.0, substrate=data_cfg.substrate, structure=data_cfg.structure)
    with torch.no_grad():
        scatterings = torcwa_simulation(phy_kwargs, layer=layer, rcwa_orders=data_cfg.rcwa_orders, validity_guard=True)
        while scatterings is None: # in case the Fourier orders truncation caused the approximation to be too inaccurate, the simulation returns None and must be repeated
            layer = generate_fn_wrapper(generate_fn, **kwargs)           
            scatterings = torcwa_simulation(phy_kwargs, layer=layer, rcwa_orders=data_cfg.rcwa_orders, validity_guard=True)
    
    # Pack.
    sample['name'] = f"sample_{idx}_h{h:.3f}_lam{lam:.3f}"
    sample['layer'] = np.where(layer.cpu() > 0.5, 1.0, -1.0)  # All layer are topologies only (-1, 1)
    sample['lvec'] = np.array([lam], dtype=np.float32)
    sample['h_original'] = np.array([h], dtype=np.float32)
    sample['h'] = np.array([utils.normalize_symmetric(h, h_max, h_min)], dtype=np.float32) # deprecated attributes
    sample['Tte'] = scatterings['Tte'].detach().cpu().numpy().astype(np.float32)
    sample['Rte'] = scatterings['Rte'].detach().cpu().numpy().astype(np.float32)
    sample['Ttm'] = scatterings['Ttm'].detach().cpu().numpy().astype(np.float32)
    sample['Rtm'] = scatterings['Rtm'].detach().cpu().numpy().astype(np.float32)
    
    return sample


def get_optimized_data_sample(idx, data_cfg, lam_str, targets_dictionary, keys):

    target_name = keys[idx%len(keys)]
    target = targets_dictionary[target_name]
    
    lam = float(lam_str)
    h_min, h_max = min(data_cfg.heights), max(data_cfg.heights)
    
    valid = False
    retries_counter = 0
    while not valid:
        try:
            h = np.random.uniform(h_min, h_max)
            min_feature_size = 0.4 #if 'splitter' in target_name else 1.2 if 'prism' in target_name else 0.25
            min_feature_size *= 1+(torch.randn(1)*0.2).clamp(-1,1).mul(0.5).item() # add some randomness to the min_feature_size
            if 'prism' in target_name:
                if target_name[-1] == '0': # then the deflection is to (1,0) or (-1,0)    
                    sx = utils.sample_shifted_exponential(n_samples=1, lam=1, device=torch.device('cuda')) 
                    sy = utils.sample_shifted_exponential(n_samples=1, lam=3, device=torch.device('cuda')) # smoother horizontally
                elif target_name[-1] == '1': # then the deflection is to (0,1) or (0,-1)
                    sx = utils.sample_shifted_exponential(n_samples=1, lam=3, device=torch.device('cuda')) # smoother vertically
                    sy = utils.sample_shifted_exponential(n_samples=1, lam=1, device=torch.device('cuda'))
            else:
                sx = utils.sample_shifted_exponential(n_samples=1, lam=1, device=torch.device('cuda'))
                sy = utils.sample_shifted_exponential(n_samples=1, lam=1, device=torch.device('cuda'))
            loss_type = 'maximize+mse' if ('prism' in target_name) or ('splitter' in target_name) else 'ue+mse' if 'uniform' in target_name else 'mse'
            opt_cfg = EasyDict(lr=5e-3, num_steps=500, loss_type=loss_type, log_every=np.inf, min_feature_size=min_feature_size, x_shrink=sx, y_shrink=sy)
            sample = optimize(target, h, lam, data_cfg, opt_cfg, verbose=False, target_name=target_name)
            valid = True
            retries_counter = 0
        except RuntimeError:
            warnings.warn('\033[93m'+f'Optimization #{retries_counter} failed for {target_name} at {lam:.3f}[um] with min_feature_size {min_feature_size:.3f}[um]. Retrying...'+'\033[0m')
            retries_counter += 1
            if retries_counter > 5:
                raise RuntimeError('\033[91m'+f'Optimization failed {retries_counter} times for {target_name} at {lam:.3f}[um] with min_feature_size {min_feature_size:.3f}[um].'+'\033[0m')
            
    return sample


def generate_grating_profile(image_size, p, device=torch.device('cuda')):
    r = int(np.ceil(0.1 / (p / image_size))) # 0.1 um is the minimum feature size
    valid = False
    M = image_size
    while not valid:
        N = torch.randint(1, 6, size=(1,)) # Number of stripes
        S = torch.randperm(M)[:N] // (2*r) * (2*r)  # stripes start indices, quatized by r
        S = S.unique()
        valid = len(S) == N 

    S = torch.sort(S)[0]
    W = torch.zeros_like(S) # stripes widths
    for i in range(N):
        if i == N-1:
            W[i] = M - S[i]
        else:
            W[i] = torch.randint(r, S[i+1] - S[i], size=(1,))

    W = W // r * r
    layer = torch.zeros(M, M, device=device)
    for i in range(N):
        layer[S[i]:S[i]+W[i], :] = 1

    if torch.rand(1) < 0.5:
        layer = rotate(layer.unsqueeze(0).unsqueeze(0), 90).squeeze(0).squeeze(0)
    
    return layer

def get_prism_data_sample(idx, data_cfg, lam_str, device=torch.device('cuda')):
    sample = {}
    layer = generate_grating_profile(data_cfg, device=device)
    lam = float(lam_str)
    h_min, h_max = min(data_cfg.heights), max(data_cfg.heights)
    h = np.random.uniform(h_min, h_max)
    phy_kwargs = dict(periodicity=data_cfg.periodicity, h=h, lam=lam, tet=0.0, substrate=data_cfg.substrate, structure=data_cfg.structure)

    with torch.no_grad():
        scatterings = torcwa_simulation(phy_kwargs, layer=layer, rcwa_orders=data_cfg.rcwa_orders, validity_guard=True)
        while scatterings is None: # in case the Fourier orders truncation caused the approximation to be too inaccurate, the simulation returns None and must be repeated
            layer = generate_grating_profile(data_cfg, device=device)           
            scatterings = torcwa_simulation(phy_kwargs, layer=layer, rcwa_orders=data_cfg.rcwa_orders, validity_guard=True)
    sample['name'] = f"sample_{idx}_h{h:.3f}_lam{lam:.3f}"
    sample['layer'] = np.where(layer.cpu() > 0.5, 1.0, -1.0)  # All layer are topologies only (-1, 1)
    sample['lvec'] = np.array([lam], dtype=np.float32)
    sample['h_original'] = np.array([h], dtype=np.float32)
    sample['h'] = np.array([utils.normalize_symmetric(h, h_max, h_min)], dtype=np.float32) # deprecated attributes
    sample['Tte'] = scatterings['Tte'].detach().cpu().numpy().astype(np.float32)
    sample['Rte'] = scatterings['Rte'].detach().cpu().numpy().astype(np.float32)
    sample['Ttm'] = scatterings['Ttm'].detach().cpu().numpy().astype(np.float32)
    sample['Rtm'] = scatterings['Rtm'].detach().cpu().numpy().astype(np.float32)

    return sample

def get_last_idx(env):
        last_idx = -1  # Initialize to -1 in case LMDB is empty
        with env.begin() as txn:
            cursor = txn.cursor()
            # Iterate over all keys to find the last one
            for key, _ in cursor:
                if int(key.decode('ascii')) > last_idx:
                    last_idx = int(key.decode('ascii'))
        return last_idx

def save_batch_to_lmdb(batch, start_idx, env, compress=True):
    with env.begin(write=True) as txn:
        for i, sample in enumerate(batch):
            serialized_data = pickle.dumps(sample)
            if compress:
                serialized_data = zlib.compress(serialized_data)
            txn.put(f"{start_idx + i}".encode('ascii'), serialized_data)


def main(args):

    data_cfg = data_config.get_data_cfg(args.data_cfg)
    lmdb_root_path = join(DATA_DIR, data_cfg.lmdb_root_path.test if args.test else data_cfg.lmdb_root_path.train)
    if args.optimized:
        lmdb_root_path += '_optimized'
    elif args.prism:
        lmdb_root_path += '_prism'
    os.makedirs(lmdb_root_path, exist_ok=True)

    device = torch.device(args.device)

    # Generate and save by batches.
    for lam_str in data_cfg.wavelengths[args.start_at:args.end_at]:
        
        # Detect whether the LMDB file already exists, and if so, whether it should be overwritten.
        env_path = join(lmdb_root_path, f'lmdb_dataset_lam{lam_str}')
        if os.path.exists(env_path) and args.overwrite:
            if input(f'[lam = {lam_str}] Data is about to be lost and overwritten. Are you sure? (y/n)\n') == 'y':
                shutil.rmtree(lmdb_root_path)
            else:
                print('Exiting...')
                return
        elif os.path.exists(env_path):
            env = lmdb.open(join(lmdb_root_path, f'lmdb_dataset_lam{lam_str}'), map_size=int(3*1024**3) ,writemap=True) # don't change map size of existing envs
            print(f'Appending to existing LMDB environment at {env_path} with map size {env.info()["map_size"]/(1024**3):.2f}GB')
        else:     
            map_size = int(3*1024**3)
            print(f'Starting a new LMDB environment at {env_path} with map size {map_size/(1024**3):.2f}GB')
            os.makedirs(env_path, exist_ok=False)
            env = lmdb.open(join(lmdb_root_path, f'lmdb_dataset_lam{lam_str}'), map_size=map_size, writemap=True)

        global_idx = get_last_idx(env) + 1

        if args.optimized:
            d = get_base_polarized_target_patterns(data_cfg)
            keys = list(d.keys())

        for start_idx in range(global_idx, args.n_structures, args.save_every):
            end_idx = min(start_idx + args.save_every, args.n_structures)
            if args.optimized:
                batch = Parallel(n_jobs=args.n_jobs)(
                    delayed(get_optimized_data_sample)(idx, data_cfg, lam_str, d, keys)
                    for idx in tqdm(range(start_idx, end_idx), f'[lam={lam_str}] Optimizing {start_idx}:{end_idx}')
                )
            elif args.prism:
                batch = Parallel(n_jobs=args.n_jobs)(
                    delayed(get_prism_data_sample)(idx, data_cfg, lam_str, device)
                    for idx in tqdm(range(start_idx, end_idx), f'[lam={lam_str}] Prisming {start_idx}:{end_idx}')
                )
            else:
                batch = Parallel(n_jobs=args.n_jobs)(
                        delayed(get_data_sample)(idx, data_cfg, lam_str, device)
                        for idx in tqdm(range(start_idx, end_idx), f'[lam={lam_str}] Generating {start_idx}:{end_idx}')
                )

            save_batch_to_lmdb(batch, start_idx, env, compress=True)
            torch.cuda.empty_cache()

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str, default='x3')
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--start_at", type=int, default=0)
    parser.add_argument("--end_at", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=12)
    parser.add_argument("--n_structures", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--overwrite", action='store_true', default=False)
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--optimized", action='store_true', default=False)
    parser.add_argument("--prism", action='store_true', default=False)

    args = parser.parse_args()
    main(args)

