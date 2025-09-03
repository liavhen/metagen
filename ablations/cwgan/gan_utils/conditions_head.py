import torch
from diffusion.edm_networks import *


# Adapted from EDM (fron SongUNet): https://github.com/NVlabs/edm/blob/main/training/networks.py

class ConditionsHead(torch.nn.Module):
    
    def __init__(self,
        label_dim              ,            # Number of class labels, 0 = unconditional.
        out_dim                ,          # Base multiplier for the number of channels.
        # channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
        # channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
    ):
        
        super().__init__()
        self.label_dropout = label_dropout
        init = dict(init_mode='xavier_uniform')

        # Mapping.
        self.map_label = Linear(in_features=label_dim, out_features=label_dim, **init)
        self.map_layer0 = Linear(in_features=label_dim, out_features=2*label_dim, **init)
        self.map_layer1 = Linear(in_features=2*label_dim, out_features=out_dim, **init)


    def forward(self,class_labels):
        tmp = class_labels
        if self.training and self.label_dropout:
            tmp = tmp * (torch.rand([class_labels.shape[0], 1], device=class_labels.device) >= self.label_dropout).to(tmp.dtype)
        emb = class_labels + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))
        return emb