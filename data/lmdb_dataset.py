from utils.paths import *
from os.path import join
from utils import utils
import torch
from torchvision.transforms import ToTensor, Resize, InterpolationMode
from torch.utils.data import ConcatDataset, DataLoader, random_split, Dataset
from scipy import io as sio
import lmdb
import pickle
import zlib  # For optional decompression
from data.data_config import *


class MetaLensDatasetLMDB(Dataset):
    def __init__(
            self,
            data_cfg,                   # data configuration from data.data_config.py
            test=False,                 # If True, load test dataset, otherwise train dataset   
            override_root=None,         # Specify root path to LMDB environment
            override_wavelengths=None,  # Specify wavelengths to include instead of those in data_cfg
            override_heights=None,      # Specify range of heights as [h_min, h_max] to restrict instead of those in data_cfg - UNSUPPORTED
            compress=True,              # If lmdb environment contains compressed files, set True, otherwise False.
            initial_scale=None,         # Resolution to which the layer components of the samples are resized
            size_limit=None,            # Limit the size of the dataset
            augments=True,              # Whehter to apply augmentations. The available augmentations are cyclic shift and random masking of Tte/Rte/Ttm/Rtm.
            max_masked=None             # Maximum number of components (T/R x TE/TM) to mask (no more than 3)
    ):

        self.data_cfg = data_cfg
        self.label_dim = utils.get_label_dim(self.data_cfg)
        self.test = test

        assert 'lmdb_root_path' in data_cfg.__dict__.keys(), "Data configuration must contain 'lmdb_root_path' key."
        if override_root is not None:
            self.lmdb_root_path = override_root
        else:
            self.lmdb_root_path = self.data_cfg.lmdb_root_path.test if self.test else self.data_cfg.lmdb_root_path.train
        self.wavelengths = override_wavelengths if override_wavelengths is not None else self.data_cfg.wavelengths
        self.heights = override_heights if override_heights is not None else self.data_cfg.heights

        self.allowed_suffices = []
        for lam in self.wavelengths:
            self.allowed_suffices.append(f'lam{lam}')
            # for h in self.heights:
                # self.allowed_suffices.append(f'h{h}_lam{lam}')
        self.envs = [lmdb.open(join(self.lmdb_root_path, lmdb_path), readonly=True, lock=False) for lmdb_path in os.listdir(self.lmdb_root_path) if lmdb_path[13:] in self.allowed_suffices]
        self.txns = [env.begin(write=False) for env in self.envs]
        self.keys = []

        # Populate keys and index mapping
        for i, txn in enumerate(self.txns):
            with txn.cursor() as cursor:
                for n, (key, _) in enumerate(cursor):
                    if n == size_limit:
                        break
                    self.keys.append((i, key))


        # Data-handling
        self.totensor = ToTensor()
        self.augments = augments
        self.max_masked = max_masked if max_masked is not None else sum(utils.get_components_booleans(self.data_cfg)) - 1
        if self.augments:
            assert self.max_masked < sum(utils.get_components_booleans(self.data_cfg)), "max_masked must be smaller than the number of included components."
                
        
        temp = {'root': self.lmdb_root_path, 'size': len(self.keys), 'label_dim': self.label_dim, 'max_masked': 'None' if not self.augments else self.max_masked, 'wavelengths': self.wavelengths, 'heights': self.heights, 'size_limit': size_limit}
        self.identity_string =  ''.join([f'\n  (*) {k:12s}: {v}' for k, v in temp.items()])

        self.res = data_cfg.resolution if initial_scale is None else initial_scale
        self.compress = compress

        if initial_scale is not None:
            self.resize = Resize((initial_scale, initial_scale), interpolation=InterpolationMode.NEAREST_EXACT)
        else:
            self.resize = None
        
    def __repr__(self):
        return f"MetaLensDatasetLMDB - {self.data_cfg.name.upper()} - {'Test' if self.test else 'Train'} - {self.identity_string}"
    
    def __len__(self):
        return len(self.keys)

    def __del__(self):
        """Close all LMDB environments on deletion."""
        for env in self.envs:
            env.close()

    def cyc_shift(self, sample_dict):
        layer = sample_dict['layer'].clone()
        pxl_res = layer.shape[-1]
        possible_shifts = torch.arange(pxl_res)
        h_shift = possible_shifts[torch.randperm(len(possible_shifts))][0].item()
        w_shift = possible_shifts[torch.randperm(len(possible_shifts))][0].item()
        rnd_shifts = (h_shift, w_shift)
        layer = torch.roll(layer, rnd_shifts, dims=(-2, -1))
        sample_dict['layer'] = layer
        return sample_dict, rnd_shifts

    def get_layer(self, sample):
        layer = self.totensor(sample['layer']).to(torch.float32)
        assert not layer.isnan().any(), "Tensor contains NaN values!"
        layer = utils.normalize01(layer) if len(layer.unique()) > 1 else layer / layer.max() if layer.max() > 0 else layer
        layer = self.resize(layer) if self.resize is not None else layer
        h0 = self.get_scalar(sample, 'h_original')
        h = h0 / max([float(h) for h in self.heights]) * torch.ones_like(layer)
        return torch.cat((layer, h), dim=-3)

    def get_scattering(self, sample):
        try:
            Tte = utils.crop_around_center(torch.tensor(sample['Tte'], dtype=torch.float32), self.data_cfg.info_t_orders).flatten()
            Rte = utils.crop_around_center(torch.tensor(sample['Rte'], dtype=torch.float32), self.data_cfg.info_r_orders).flatten()
            Ttm = utils.crop_around_center(torch.tensor(sample['Ttm'], dtype=torch.float32), self.data_cfg.info_t_orders).flatten()
            Rtm = utils.crop_around_center(torch.tensor(sample['Rtm'], dtype=torch.float32), self.data_cfg.info_r_orders).flatten()
        except KeyError as e:
            # handles old dataset (before the different polarizations were added)
            Tte = utils.crop_around_center(torch.tensor(sample['T'], dtype=torch.float32), self.data_cfg.info_t_orders).flatten()
            Rte = utils.crop_around_center(torch.tensor(sample['R'], dtype=torch.float32), self.data_cfg.info_r_orders).flatten()
            Ttm = torch.zeros_like(Tte)
            Rtm = torch.zeros_like(Rte)
            
        tte, rte, ttm, rtm = utils.get_components_booleans(self.data_cfg)
        
        if self.augments:
            mask = utils.get_random_scattering_mask(self.data_cfg, max_masked=self.max_masked, device=Tte.device)
        else:
            mask = torch.ones(4)
        
        included_components = tte*[mask[0]*Tte] + rte*[mask[1]*Rte] + ttm*[mask[2]*Ttm] + rtm*[mask[3]*Rtm]

        scattering = torch.cat(included_components + [self.get_scalar(sample, 'lvec').view(-1)], dim=-1)
        assert scattering.shape[-1] == self.label_dim, f"Scattering tensor has shape {scattering.shape}, but expected {self.label_dim}."
        
        return scattering, mask

    def get_scalar(self, dict, key):
        return torch.tensor(dict[key][0], dtype=torch.float32)

    def __getitem__(self, idx):
        env_index, key = self.keys[idx]
        txn = self.txns[env_index]
        with txn.cursor() as cursor:
            data_bytes = cursor.get(key)
        if data_bytes is None:
            raise IndexError(f"Key {key} not found in the LMDB database {env_index}.")

        if self.compress:
            data_bytes = zlib.decompress(data_bytes)

        sample = pickle.loads(data_bytes)

        sample_dict = dict()
        sample_dict['name'] = sample['name']
        sample_dict['layer'] = self.get_layer(sample)
        sample_dict['scattering'], sample_dict['mask'] = self.get_scattering(sample)
        sample_dict['h'] = self.get_scalar(sample, 'h')
        sample_dict['h_original'] = self.get_scalar(sample, 'h_original')
        sample_dict['h_max'] = max(self.heights)
        sample_dict['h_min'] = min(self.heights)
        sample_dict['lvec'] = self.get_scalar(sample, 'lvec')
        sample_dict['per'] = self.data_cfg.periodicity

        # Augmentations
        rnd_shift = (0, 0)
        if self.augments:
            sample_dict, rnd_shift = self.cyc_shift(sample_dict)
        sample_dict['augments'] = rnd_shift

        return sample_dict


if __name__ == "__main__":
    from data.pca import PCA
    from evaluation.quality_evaluation import get_special_conditions
    from torchvision.utils import save_image
    from matplotlib import pyplot as plt

    N = 5000

    ds = MetaLensDatasetLMDB(size_limit=100)
    batch = DataLoader(ds, batch_size=1000, shuffle=True).__iter__().__next__()
    s = batch['scattering'].cuda()

    target, _ = get_special_conditions(torch.device('cuda'), wavelengths=ALL_WAVELENGTHS)
    all_conditions = torch.cat([s, target], dim=0)
    pca = PCA(n_components=2).cuda().fit(s)
    t = pca.transform(all_conditions)
    pca1 = t[:N, 0].cpu().numpy()
    pca2 = t[:N, 1].cpu().numpy()
    plt.scatter(pca1, pca2, marker='.', label='new distribution', alpha=0.2)
    pca1 = t[N:, 0].cpu().numpy()
    pca2 = t[N:, 1].cpu().numpy()
    plt.scatter(pca1, pca2, marker='.', label='target distribution', color='r')

    plt.savefig(join(PROJECT_DIR, 'data', 'figs', f'new_data_distribution_tag.jpg'))

    mini_batch_layers = batch['layer'][:100]
    save_image(utils.viewable(mini_batch_layers), join(PROJECT_DIR, 'data', 'figs', f'mini_batch_layers_tag.jpg'), nrow=10, pad_value=1)
