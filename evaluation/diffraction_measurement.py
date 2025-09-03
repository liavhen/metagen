import torch
import torcwa
import matplotlib.pyplot as plt
import utils
import numpy as np
from torchvision.transforms import Resize, functional
from utils.utils import normalize01
from utils.paths import *
from os.path import join
from data.material import Material
import warnings


def torcwa_simulation(phy_kwargs, layer, rcwa_orders=7, validity_guard=False, project=True):
    """
        :param phy_kwargs: TODO: update the expected args in phy_kwargs
            periodicity: meta-atom periodicity in microns
            h: layer's thickness (or height) in microns
            lam: lambda, wavelength of the incident wave, in microns
            tet: theta, angle of the incident planar wave, in degrees
            substrate: substrate material (e.g. 'SiO2')
            structure: structure material (e.g. 'Si')
        :param layer: image of the layer (rectangles will be extracted from it if provided)
        :param rcwa_orders: number of orders in the RCWA simulation
        :param validity_guard: whether to return None if the energy conservation is violated
        :param project: whether to project the layer to binary values or not

        :return: scattering (dict): scattering efficiencies represented as matrices (for both TE and TM polarizations)

    """

    # Build & Run RCWA Simulation
    torch.backends.cuda.matmul.allow_tf32 = False
    torcwa.rcwa_geo.Lx = phy_kwargs['periodicity'] * 1000  # nm
    torcwa.rcwa_geo.Ly = phy_kwargs['periodicity'] * 1000  # nm
    torcwa.rcwa_geo.nx = layer.shape[0]
    torcwa.rcwa_geo.ny = layer.shape[1]
    torcwa.rcwa_geo.grid()
    torcwa.rcwa_geo.edge_sharpness = 10000000.

    sim_dtype = torch.complex64

    # order = [11, 11]
    order = [rcwa_orders, rcwa_orders]
    L = [torcwa.rcwa_geo.Lx, torcwa.rcwa_geo.Ly]
    torch.set_num_threads(1)
    sim = torcwa.rcwa(freq=1 / (phy_kwargs['lam'] * 1000), order=order, L=L, dtype=sim_dtype, device=layer.device, stable_eig_grad=False)

    # Materials
    substrate_eps = Material.apply(phy_kwargs['substrate'], phy_kwargs['lam']) ** 2
    structure_eps = Material.apply(phy_kwargs['structure'], phy_kwargs['lam']) ** 2
    air_eps = 1.0

    # Build device properties
    beta = 10
    if project:
        layer = utils.binary_projection(layer, beta) if layer.requires_grad else utils.threshold(layer, 0.5)  # input layer is of range (0,1) with noise, and must be binarized
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=phy_kwargs['tet'], azi_ang=0)
    sim.add_layer(thickness=phy_kwargs['h'] * 1000., eps=(structure_eps * layer + air_eps * (1. - layer)))

    # Solve RCWA
    sim.solve_global_smatrix()

    # Extract efficiencies ((-9, 9) x (9, -9)
    o = torch.arange(-19 // 2, 19 // 2) + 1
    all_orders = torch.stack(torch.meshgrid(o, o, indexing='ij'), dim=-1).reshape(-1, 2)

    # TE
    tss = sim.S_parameters(orders=all_orders, direction='f', port='t', polarization='ss')
    tps = sim.S_parameters(orders=all_orders, direction='f', port='t', polarization='ps')
    rss = sim.S_parameters(orders=all_orders, direction='f', port='r', polarization='ss')
    rps = sim.S_parameters(orders=all_orders, direction='f', port='r', polarization='ps')
    Tte = torch.abs(tss) ** 2 + torch.abs(tps) ** 2
    Rte = torch.abs(rss) ** 2 + torch.abs(rps) ** 2

    # TM
    tpp = sim.S_parameters(orders=all_orders, direction='f', port='t', polarization='pp')
    tsp = sim.S_parameters(orders=all_orders, direction='f', port='t', polarization='sp')
    rpp = sim.S_parameters(orders=all_orders, direction='f', port='r', polarization='pp')
    rsp = sim.S_parameters(orders=all_orders, direction='f', port='r', polarization='sp')
    Ttm = torch.abs(tpp) ** 2 + torch.abs(tsp) ** 2
    Rtm = torch.abs(rpp) ** 2 + torch.abs(rsp) ** 2

    tolerance = 5e-2
    if torch.abs(torch.sum(Tte+Rte) - 1.0) > tolerance or torch.abs(torch.sum(Ttm+Rtm) - 1.0) > tolerance:
        warnings.warn(f"Energy Conversion is violated by more than {tolerance*100:.2f}% (TE/TM Total Energy = {torch.sum(Tte+Rte).item():.3f}/{torch.sum(Ttm+Rtm).item():.3f})! Check your simulation.")
        if validity_guard:
            return None

    output = {
        'Tte': Tte.view(19, 19),
        'Rte': Rte.view(19, 19),
        'Ttm': Ttm.view(19, 19),
        'Rtm': Rtm.view(19, 19),
        'all': torch.cat([Tte, Rte, Ttm, Rtm], dim=0)
    }

    return output

