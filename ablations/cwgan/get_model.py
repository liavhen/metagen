from ablations.cwgan.models.wgan_gradient_penalty import WGAN_GP
from ablations.cwgan.models.wgan_clipping import WGAN_CP
from ablations.cwgan.models.gan import GAN
from ablations.cwgan.models.dcgan import DCGAN_MODEL
from edm_utils.dnnlib.util import EasyDict
from os.path import join


def get_model(
        data_cfg, 
        model_type      = "WGAN-GP",        # Model Type - Must be supported in cwgan/models/*
        latent_dim      = 100,              # Dimensionality of the latent space
        img_channels    = 2,                # Number of input image channels
        label_dim       = 0,                # Number of class labels, 0 = unconditional
        batch_size      = 64,               # The size of batch
        cuda            = True,             # Availability of cuda
        generator_iters = 10000,            # The number of iterations for generator in WGAN model
        critic_iters    = 1,                # The number of critic iterations for every iteration of the generator in WGAN model
        ckpts_path      = None,             # Path to checkpoints (directory containing 'generator.pkl' and 'discriminator.pkl')
        random_init     = False,            # Get randomly initialized model
        **enexpected_kwargs
):
# atent_dim, img_channels, ≈, args):
    
    args = EasyDict(
        latent_dim=latent_dim, 
        img_channels=img_channels, 
        label_dim=label_dim,
        batch_size=batch_size,
        cuda= cuda=='True',
        generator_iters=generator_iters,
        critic_iters=critic_iters
    )

    if model_type == 'WGAN-GP':
        model = WGAN_GP(data_cfg=data_cfg, args=args)
    elif model_type == 'WGAN-CP':
        model = WGAN_CP(data_cfg=data_cfg, args=args)
    elif model_type == 'GAN':
        model = GAN(data_cfg=data_cfg, args=args)
    elif model_type == 'DCGAN':
        model = DCGAN_MODEL(data_cfg=data_cfg, args=args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    if not random_init:
        ckpts_path = ckpts_path if ckpts_path is not None else data_cfg.ckpts_path.cwgan
        model.load_pretrained(ckpts_path, resume=False)

    return model