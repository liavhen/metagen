import torch
from ablations.vae_lib.models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from ablations.vae_lib.vae_utils import ConditionsHead


class ResidualLayer(nn.Module):
    # Same as provided code
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)
    

class ModifiedConditionalVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size:int = 64,
                 **kwargs) -> None:
        super(ModifiedConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.in_channels = in_channels
        self.label_dim = num_classes

        # self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Add conditioning channel to every h_dim in hidden_dims but the first and the last
        # hidden_dims = [in_channels + 1] + [h_dim + 1 for i, h_dim in enumerate(hidden_dims)]
        # in_channels += 1 # To account for the extra label channel


        # Build Encoder
        modules = []
        encoder_embedders = []

        for h_dim in hidden_dims:
            encoder_embedders.append(ConditionsHead(self.label_dim, img_size**2))
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + 1, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )

            img_size = img_size // 2
            in_channels = h_dim
        
        for i in range(3):
            encoder_embedders.append(None)
            modules.append(ResidualLayer(in_channels, in_channels)) 

        self.encoder = nn.Sequential(*modules)
        self.encoder_embedders = nn.Sequential(*encoder_embedders)
        
        self.enc_final_label_embedder = ConditionsHead(self.label_dim, latent_dim)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4 + latent_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4 + latent_dim, latent_dim)


        # Build Decoder
        modules = []
        decoder_embedders = []

        self.dec_init_label_embedder = ConditionsHead(self.label_dim, latent_dim)
        self.decoder_input = nn.Linear(2* latent_dim, hidden_dims[-1] * 4)

        for i in range(3):
            decoder_embedders.append(None)
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            decoder_embedders.append(ConditionsHead(self.label_dim, img_size**2))
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i] + 1,
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

            img_size = img_size * 2

        self.decoder = nn.Sequential(*modules)
        self.decoder_embedders = nn.Sequential(*decoder_embedders)

        self.dec_final_label_embedder = ConditionsHead(self.label_dim, img_size**2)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1] + 1,
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, x: Tensor, labels: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x:       (Tensor)  Input tensor to encoder [N x C x H x W]
        :param labels:  (Tensor) Labels tensor to encoder [N x self.num_classes]
        :return:        (Tensor) List of latent codes
        """
        
        for i, module in enumerate(self.encoder):
            if isinstance(module, ResidualLayer):
                x = module(x)
            else:
                y = self.encoder_embedders[i](labels)
                _x = torch.cat([x, y.reshape(-1, 1, x.size(2), x.size(3))], dim = 1)
                x = module(_x)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        enc_label_emb = self.enc_final_label_embedder(labels)
        result = torch.cat([x.flatten(start_dim=1), enc_label_emb], dim = 1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        emb = self.dec_init_label_embedder(labels)
        x = self.decoder_input(torch.cat([z, emb], dim = 1))
        x = x.view(-1, 512, 2, 2)
        for i, module in enumerate(self.decoder):
            if isinstance(module, ResidualLayer):
                x = module(x)
            else:
                y = self.decoder_embedders[i](labels)
                _x = torch.cat([x, y.reshape(-1, 1, x.size(2), x.size(3))], dim = 1)
                x = module(_x)
        
        final_emb = self.dec_final_label_embedder(labels)
        result = self.final_layer(torch.cat([x, final_emb.reshape(-1, 1, x.size(2), x.size(3))], dim = 1))
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels'].float()
        # embedded_class = self.embed_class(y)
        # embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        x = self.embed_data(input)

        # x = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x, y)

        z = self.reparameterize(mu, log_var)

        # z = torch.cat([z, y], dim = 1)
        return  [self.decode(z, y), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        # z = torch.cat([z, y], dim=1)
        samples = self.decode(z, y)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]