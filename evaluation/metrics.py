import torch
import numpy as np


def l1norm(x):
    return np.sum(np.abs(x))


def batchsum(x):
    return torch.sum(x, dim=tuple(range(1, x.dim())))


def relative_error(x_hat, x_gt):
    try:
        x_hat = x_hat.to(x_gt.device)
        sum_fn = torch.sum if x_hat.dim() == 2 and x_hat.shape[-1] == x_hat.shape[-2] else batchsum
        return sum_fn(torch.abs(x_hat - x_gt)) / sum_fn(torch.abs(x_gt))
    except RuntimeError:
        return torch.tensor([torch.nan], device=x_gt.device)


def uniformity_error(x_hat, x_gt):
    try:
        if x_hat.dim() == 2 and x_hat.shape[-1] == x_hat.shape[-2]:
            _x = x_hat[x_gt != 0]
            x_max, x_min = torch.max(_x, dim=-1).values, torch.min(_x, dim=-1).values
        else:
            _x = [x_hat[i, x_gt[i] != 0] for i in range(x_hat.shape[0])]
            x_max = torch.stack([_x[i].max() for i in range(x_hat.shape[0])], dim=0)
            x_min = torch.stack([_x[i].min() for i in range(x_hat.shape[0])], dim=0)
        return (x_max - x_min) / (x_max + x_min)
    except RuntimeError:
        return torch.tensor([torch.nan], device=x_gt.device)


def nrms_error(x_hat, x_gt):
    try:
        if x_hat.dim() == 1:
            return ((x_hat - x_gt) / (x_gt + 1e-4)).pow(2).mean().sqrt().to(torch.float32)
        elif x_hat.dim() == 2:
            if x_hat.shape[-1] == x_hat.shape[-2]:
                return ((x_hat - x_gt) / (x_gt + 1e-4)).pow(2).mean().sqrt().to(torch.float32)
            else:
                return ((x_hat - x_gt) / (x_gt + 1e-4)).pow(2).mean(dim=-1).sqrt().to(torch.float32)
        else:
            return ((x_hat - x_gt) / (x_gt + 1e-4)).pow(2).mean(dim=(-2, -1)).sqrt().to(torch.float32)
    except RuntimeError:
        return torch.tensor([torch.nan], device=x_gt.device)
