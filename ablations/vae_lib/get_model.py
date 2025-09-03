from ablations.vae_lib.models import vae_models
import torch

def get_model(data_cfg, **model_kwargs):

    model_type = model_kwargs['model_type']
    config = {
        'in_channels': model_kwargs['img_channels'],
        'num_classes': model_kwargs['label_dim'],
        'latent_dim': model_kwargs['latent_dim'],
        'img_size': model_kwargs['img_resolution'],
    }
    model = vae_models[model_type](**config)

    try:
        ckpts_path = model_kwargs['ckpts_path']
    except KeyError:
        ckpts_path = data_cfg.ckpts_path.cvae
        
    ckpts = torch.load(ckpts_path, weights_only=False)
    state_dict = {k[6:]: v for k, v in ckpts['state_dict'].items()} # workaround for migrating lightning checkpoints
    model.load_state_dict(state_dict)
    return model


