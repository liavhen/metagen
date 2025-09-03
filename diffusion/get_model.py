import torch
from diffusion.edm_networks import EDMPrecond
from torch.nn.parallel import DataParallel
from utils.utils import state_dict_from_parallel_to_single


def get_model(
        data_cfg,                           # Data configuration
        label_dim       = 4*19**2+1,        # Number of class labels, 0 = unconditional
        img_channels    = 2,                # Number of input image channels
        parallel        = False,            # Use DataParallel or not
        model_type      = "SongUNet",       # Model Type - Must be supported in model/edm_networks.py
        model_channels  = 128,              # Base number of channels in the network
        num_blocks      = 4,                # Number of residual blocks per resolution
        label_dropout   = 0.1,              # label dropout support (for classifier-free guidance)
        dropout         = 0.1,              # Dropout for weights regularization
        ckpts_path      = None,             # Path to checkpoints
        random_init     = False,            # Get randomly initialized model
        **unexpected_kwargs

):


    model_kwargs = dict(
        model_type=model_type,
        model_channels=model_channels,
        num_blocks=num_blocks,
        label_dropout=label_dropout,
        dropout=dropout,
    )

    # extra args required for DiT init
    if model_type == 'DiTL8':
        model_kwargs.update(model_channels=1024, depth=24, num_heads=16, patch_size=8, learn_sigma=False)
    elif model_type == 'DiTB8':  # extra args required for DiT init
        model_kwargs.update(model_channels=768, depth=12, num_heads=12, patch_size=8, learn_sigma=False)
    if model_type == 'DiTS8':  # extra args required for DiT init
        model_kwargs.update(model_channels=384, depth=12, num_heads=6, patch_size=8, learn_sigma=False)

    interface_kwargs = dict(img_resolution=data_cfg.resolution, img_channels=img_channels, label_dim=label_dim)

    model = EDMPrecond(**interface_kwargs, **model_kwargs)
    model = DataParallel(model) if parallel else model
    
    if not random_init:
        ckpts_path = ckpts_path if ckpts_path is not None else data_cfg.ckpts_path.metagen
        checkpoint = torch.load(ckpts_path, map_location=lambda storage, loc: storage, weights_only=False)
        if not parallel:
            checkpoint = state_dict_from_parallel_to_single(checkpoint)
        model.load_state_dict(checkpoint['ema'])

    return model
