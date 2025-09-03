import torch
from ablations.vae_lib.models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from ablations.vae_lib.vae_utils import ConditionsHead

class VectorQuantizer(nn.Module):
    # Same as provided code
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        # Same as provided implementation
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

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

# class ConditionHead(nn.Module):
#     """
#     Processes the conditional signal using a linear layer, non-linear activation,
#     and another linear layer. The output dimension varies depending on where it is injected.
#     """
#     def __init__(self, in_features: int, out_features: int):
#         super(ConditionHead, self).__init__()
#         self.fc1 = nn.Linear(in_features, in_features)
#         self.activation = nn.ReLU()
#         self.fc2 = nn.Linear(in_features, out_features)

#     def forward(self, label: Tensor) -> Tensor:
#         return self.fc2(self.activation(self.fc1(label))).unsqueeze(-1).unsqueeze(-1)

class ConditionalVQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 label_dim: int = 10,  # Dimension of conditional label
                 label_embedding_dim: int = 128,
                 **kwargs) -> None:
        super(ConditionalVQVAE, self).__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.label_dim = label_dim
        # self.label_embedding_dim = label_embedding_dim
        self.img_size = img_size

        # self.conditions_head = ConditionsHead(self.label_dim, self.label_embedding_dim)

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]
       
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

        encoder_embedders.append(ConditionsHead(self.label_dim, img_size**2))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels + 1, in_channels,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        for i in range(6):
            encoder_embedders.append(None)
            modules.append(ResidualLayer(in_channels, in_channels)) #if i < 5 else nn.Sequential(
                # ResidualLayer(in_channels, in_channels), nn.LeakyReLU()
            # ))
      
        encoder_embedders.append(ConditionsHead(self.label_dim, img_size**2))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels + 1, self.embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        assert(len(encoder_embedders) == len(modules)), "Encoder embedders and modules must be of the same length"
        self.encoder = nn.Sequential(*modules)
        self.encoder_embedders = nn.Sequential(*encoder_embedders)


        self.vq_layer = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.beta)

        # Build Decoder

        modules = []
        decoder_embedders = []

        decoder_embedders.append(ConditionsHead(self.label_dim, img_size**2))
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim + 1,
                        hidden_dims[-1],
                        kernel_size=3,
                        stride=1,
                        padding=1),
                nn.LeakyReLU())
        )

        for i in range(6):
            decoder_embedders.append(None)
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))# if i < 5 else nn.Sequential(
                # ResidualLayer(hidden_dims[-1] + 1, hidden_dims[-1]), nn.LeakyReLU()
            # ))

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            decoder_embedders.append(ConditionsHead(self.label_dim, img_size**2))
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i] + 1,
                            hidden_dims[i + 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1),
                    nn.LeakyReLU())
            )
            img_size = img_size * 2
        
        decoder_embedders.append(ConditionsHead(self.label_dim, img_size**2))
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1] + 1,
                        out_channels=self.in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                nn.LeakyReLU())
        )
        
        assert (len(decoder_embedders) == len(modules)), "Decoder embedders and modules must be of the same length"
        self.decoder = nn.Sequential(*modules)
        self.decoder_embedders = nn.Sequential(*decoder_embedders)
        
        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
        #                               kernel_size= 3, padding= 1),
        #                     nn.Tanh())
        
        
        # self.decoder = nn.Sequential(*modules)

    def inject_condition(self, x: Tensor, condition: Tensor) -> Tensor:
        # condition = condition.expand_as(x)
        return torch.cat([x, condition], dim=1)

    def encode(self, input: Tensor, labels: Tensor) -> List[Tensor]:
        result = input
        for i, module in enumerate(self.encoder):
            if isinstance(module, ResidualLayer):
                result = module(result)
            else:
                embedded_labels = self.encoder_embedders[i](labels).reshape(result.size(0), -1, result.size(2), result.size(3))
                assert embedded_labels.size(1) == 1, "Labels embdding must have a single channel after reshaping."
                input = torch.cat([result, embedded_labels], dim=1)
                result = module(input)
        return [result]

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        result = z
        for i, module in enumerate(self.decoder):
            if isinstance(module, ResidualLayer):
                result = module(result)
            else:
                embedded_labels = self.decoder_embedders[i](labels).reshape(result.size(0), -1, result.size(2), result.size(3))
                assert embedded_labels.size(1) == 1, "Labels embdding must have a single channel after reshaping."
                input = torch.cat([result, embedded_labels], dim=1)
                result = module(input)
        return result

    def forward(self, input: Tensor, labels: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input, labels)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs, labels), input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss': vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

