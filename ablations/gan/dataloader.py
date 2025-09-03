# Modified based on https://github.com/znxlwm/pytorch-generative-model-collections

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.append('../..')
from data.dataset import MetaLensDataset
from data.dataset import collate_fn_replace_corrupted
import functools


def dataloader(dataset, input_size, batch_size, split='train', size_limit=None):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)

    elif dataset == 'MetaLensDataset':
        dataset = MetaLensDataset(split, size_limit=size_limit, augments={"cyc_shift": True})
        collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=14, collate_fn=collate_fn,
        )

    return data_loader