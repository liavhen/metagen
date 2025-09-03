import pytorch_lightning as pl
 

## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##
  

def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


import torch
from diffusion.edm_networks import *


# Adapted from EDM (fron SongUNet): https://github.com/NVlabs/edm/blob/main/training/networks.py
class ConditionsHead(torch.nn.Module):
    
    def __init__(self,
        label_dim              ,            # Number of class labels, 0 = unconditional.
        out_dim                ,          # Base multiplier for the number of channels.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
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