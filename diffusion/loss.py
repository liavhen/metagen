# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch

import utils


#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, pca=None, target=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        if pca is not None:
            assert target is not None
        self.pca = pca
        self.target = target
        self.decay_rate = 1.5

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        y, augment_labels = (augment_pipe(images), None) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        # D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        D_yn = net(y + n, sigma, labels, augment_labels=None)
        D_yn[:, 0:1, ...] = utils.align_cyclic_shift(cyc_shifted_img=D_yn[:, 0:1, ...], aligned_img=y[:, 0:1, ...]) # align first channel only
        loss = weight * ((D_yn - y) ** 2)

        if self.pca is not None:
            projected_labels = self.pca.transform(labels)
            d_from_target = torch.square(projected_labels[:, None, :] - self.target[None, :, :]).sum(dim=-1).sqrt().min(dim=-1).values
            weights_by_target = torch.cos(torch.pi * d_from_target / 2 / self.decay_rate)
            loss = weights_by_target.reshape(-1, 1, 1, 1) * loss


        # variance_weight = len(labels) * torch.softmax(labels.std(dim=-1), dim=0)
        # max_variance = 0.5
        # t_efficiencies = labels[:, :9]
        # stds = t_efficiencies.std(dim=-1).clamp(max=max_variance).reshape(-1, 1, 1, 1)
        # std_weights = torch.cos(torch.pi * stds / max_variance / 2)
        # loss = loss * std_weights

        return loss.mean(), (y, n, y + n, D_yn)

# ----------------------------------------------------------------------------


class MetagenLoss:
    def __init__(self, pnn, edm_loss_fn, pnn_loss_weight):
        self.pnn = pnn.eval().requires_grad_(False)
        self.edm_loss_fn = edm_loss_fn
        self.pnn_loss_weight = pnn_loss_weight
        self.enable_pnn_loss = False

    def __call__(self, net, images, labels=None, lams=None, augment_pipe=None):
        loss, (y, n, y_plus_n, D_yn) = self.edm_loss_fn(net, images, labels, augment_pipe)
        if labels is not None and self.enable_pnn_loss:
            layer = utils.threshold(D_yn[:, 0:1, ...], thresh=0.5)
            h = D_yn[:, 1:, ...].mean(dim=(2, 3))
            lams = lams.reshape(-1, 1)
            pred = self.pnn(layer, h, lams)
            pnn_loss = torch.nn.functional.mse_loss(pred, labels[:, :-1])  # remove wavelength from the labels (last entry)
            loss = loss + self.pnn_loss_weight * pnn_loss
        return loss.mean(), (y, n, y + n, D_yn)
