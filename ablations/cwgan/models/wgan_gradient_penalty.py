import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from os.path import join
# from gan_utils.tensorboard_logger import Logger
from itertools import chain
from torchvision import utils
from utils.paths import *
from ablations.cwgan.gan_utils.conditions_head import ConditionsHead
from utils import utils
import wandb, torchvision

# COMMENTED OUT FOR SAMPLING (ELSE, A CIRCULAR IMPORT OCCURS)
# from diffusion.sample import sample, compute_actual_scatterings, compute_metrics


class Generator(torch.nn.Module):
    def __init__(self, latent_dim, label_dim, img_channels):
        super().__init__()

        # Add conditionality
        self.emb_dim = 64**2
        self.conditions_head = ConditionsHead(label_dim, out_dim=self.emb_dim)
        self.in_channels = latent_dim + self.emb_dim
        self.out_channels = img_channels


        # Filters [1024, 512, 256]
        # Input_dim = self.in_channels (latent_dim + label_dim)
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector latent_dim
                        
            nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=self.out_channels, kernel_size=4, stride=2, padding=1)
        )
        # output of main module --> Image (Cx64x64)

        self.output = nn.Tanh()

    def forward(self, z, labels):
        labels = self.conditions_head(labels).view(labels.size(0), self.emb_dim, 1, 1)
        x = torch.cat([z, labels], 1)
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, img_dim, labels_dim):
        super().__init__()

        # Add conditionality
        self.emb_dim = 64**2 ## square image size
        self.conditions_head = ConditionsHead(labels_dim, self.emb_dim) 
        self.in_channels = img_dim + 1

        # Filters [256, 512, 1024]
        # Input_dim = self.in_channels (img_dim + labels_dim)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=self.in_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, img, labels):
        assert img.size(2) == img.size(3), 'Discriminator input must be square image'
        assert img.size(3) == self.emb_dim**0.5, 'Embedding dimension must be square of image size'
        labels = self.conditions_head(labels).view(labels.size(0), 1, img.size(2), img.size(3))
        x = torch.cat([img, labels], 1)
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class WGAN_GP(object):
    def __init__(self, data_cfg, args):
        # print("WGAN_GradientPenalty init model.")
        
        self.data_cfg = data_cfg
        self.args = args        
        self.latent_dim = args.latent_dim
        self.img_channels = args.img_channels
        self.label_dim = args.label_dim

        self.G = Generator(self.latent_dim, self.label_dim, self.img_channels)
        self.D = Discriminator(self.img_channels, self.label_dim)
        self.C = self.img_channels
        
        if hasattr(self.args, 'is_train'):
            if self.args.is_train == 'True':
                outdir = join(RUNS_DIR, f'c-wgan-gp-{self.data_cfg.name}-{utils.get_timestamp()}')
                print(f"Training mode, creating outdir {outdir}")
                self.outdir = outdir
                os.makedirs(self.outdir, exist_ok=True)

        # Check if cuda is available
        self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0 # modifies to match recommended parameters for wgan-gp
        self.b2 = 0.9 # modifies to match recommended parameters for wgan-gp
        self.batch_size = args.batch_size # 64 in the paper

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # Run-time managements
        self.curr_g_iter = 0

        # Set the logger
        # self.logger = Logger('./logs')
        # self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = args.generator_iters
        self.critic_iter = args.critic_iters
        self.lambda_term = 10

        self.sigma = 0.2 # Maximal instance noise level
        self.min_rel_error = torch.inf


    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda()
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        # print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False
    
    # @torch.no_grad()
    # def evaluate_conditions(self):
    #     self.G = self.G.eval().requires_grad_(False)
    #     self.D = self.D.eval().requires_grad_(False)
    #     results = sample(
    #         model    = self,          model_kwargs    = {'model_type': 'WGAN-GP'},
    #         data_cfg = self.data_cfg,   data_kwargs     = {'data_type': 'test', 'eval_batch_size': 100},
    #     )
    #     results = compute_actual_scatterings(results, self.data_cfg, logger=None, n_jobs=1, verbose=False)
    #     metrics = compute_metrics(results)
    #     self.G = self.G.train().requires_grad_(True)
    #     self.D = self.D.train().requires_grad_(True)
    #     return metrics['relative_t_mean']

    def train(self, train_loader):
        self.t_begin = t.time()

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for _ in range(self.curr_g_iter, self.generator_iters):
            # self.curr_g_iter = g_iter
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images, labels = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                # z = torch.rand((self.batch_size, self.latent_dim, 1, 1))

                images = self.get_torch_variable(images)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator

                # Add instance noise to the real images
                images = images + (self.generator_iters - self.curr_g_iter) / self.generator_iters * self.sigma * torch.randn_like(images)

                # Train with real images
                d_loss_real = self.D(images, labels)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, self.latent_dim, 1, 1))

                fake_images = self.G(z, labels)

                # Add instance noise to the fake images
                fake_images = fake_images + (self.generator_iters - self.curr_g_iter) / self.generator_iters * self.sigma * torch.randn_like(fake_images)

                d_loss_fake = self.D(fake_images, labels)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data, labels)
                gradient_penalty.backward()


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                # print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            _, labels = self.data.__next__()
            z = self.get_torch_variable(torch.randn(self.batch_size, self.latent_dim, 1, 1))
            fake_images = self.G(z, labels)
            g_loss = self.D(fake_images, labels)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            # Saving model and sampling images every 1000th generator iterations
            if (self.curr_g_iter) % self.args.log_every == 0:
                print(f'[{utils.now()}] Generator iteration: {self.curr_g_iter:6d}/{self.generator_iters}, g_loss: {g_loss:.5f}, d_loss: {d_loss:.5f}, gp: {gradient_penalty:.5f}, Wasserstein_D: {Wasserstein_D:.5f}')

                info = {
                    'Wasserstein distance': Wasserstein_D.data,
                    'Loss D': d_loss.data,
                    'Loss G': g_cost.data,
                    'Loss GP': gradient_penalty.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data,
                }

                # Denormalize images and save them in grid 8x8
                if self.curr_g_iter % self.args.image_every == 0:
                    _, labels = self.data.__next__()
                    z = self.get_torch_variable(torch.randn(self.batch_size, self.latent_dim, 1, 1))
                    samples = self.G(z, labels)
                    viewable_samples = utils.viewable(samples)
                    # samples = samples.mul(0.5).add(0.5)
                    viewable_samples = viewable_samples.data.cpu()
                    # grid = utils.make_grid(viewable_samples)
                    torchvision.utils.save_image(viewable_samples, join(self.outdir, 'img_generator_iter_{}.png'.format(str(self.curr_g_iter).zfill(3))), nrow=8)
                    self.save_state()
                    h_std = samples[:, 1, ...].std().mean()
                    info.update({'h_std': h_std})

                
                if self.curr_g_iter % self.args.eval_every == 0:
                    rel_error = self.evaluate_conditions()
                    info.update({'Relative Error': rel_error})
                    if rel_error < self.min_rel_error:
                        self.min_rel_error = rel_error
                        self.save_state()
                
                self.log(step=self.curr_g_iter, data=info)
            
            self.curr_g_iter += 1
            
            

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_state()

    @torch.no_grad()
    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.data = self.get_infinite_batches(test_loader)
        _, labels = self.data.__next__()
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, self.latent_dim, 1, 1))
        samples = self.G(z, labels)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')


    def calculate_gradient_penalty(self, real_images, fake_images, labels):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, labels)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # flatten the gradients to it calculates norm batchwise
        gradients = gradients.view(gradients.size(0), -1)
        
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    # def save_model(self):
    #     os.rename(join(self.outdir, f'generator.pkl'), join(self.outdir, f'prev_generator.pkl'))
    #     os.rename(join(self.outdir, f'discriminator.pkl'), join(self.outdir, f'prev_discriminator.pkl'))
    #     torch.save(self.G.state_dict(), join(self.outdir, f'generator.pkl'))
    #     torch.save(self.D.state_dict(), join(self.outdir, f'discriminator.pkl'))
    #     # print('Models save to generator.pkl & discriminator.pkl ')

    # def load_model(self, D_model_path, G_model_path):
        # D_model_path = os.path.join(os.getcwd(), D_model_filename)
        # G_model_path = os.path.join(os.getcwd(), G_model_filename)
        # self.D.load_state_dict(torch.load(D_model_path, weights_only=True))
        # self.G.load_state_dict(torch.load(G_model_path, weights_only=True))
        # self.D.load_state_dict(torch.load(D_model_path))
        # self.G.load_state_dict(torch.load(G_model_path))
        # print('Generator model loaded from {}.'.format(G_model_path))
        # print('Discriminator model loaded from {}-'.format(D_model_path))

    def save_state(self):
        # if os.path.exists(join(self.outdir, 'state_dict.pth')):
            # os.rename(join(self.outdir, 'state_dict.pth'), join(self.outdir, 'prev_state_dict.pth'))
        
        torch.save({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'curr_g_iter': self.curr_g_iter
        }, join(self.outdir, f'state_dict_{self.curr_g_iter}.pth'))
    
   
    def load_pretrained(self, ckpts_path, resume=False):
        checkpoint = torch.load(ckpts_path, weights_only=False)
        # print(f"Loading pre-trained weights from {ckpts_path}")
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        if resume:
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            self.curr_g_iter = checkpoint['curr_g_iter']


    def get_infinite_batches(self, data_loader):
        while True:
            for i, sample in enumerate(data_loader):
                layer, scattering = sample['layer'].cuda(), sample['scattering'].cuda()
                # lam = scattering[:, -1:]
                # t_scattering = scattering[:, :self.data_cfg.info_t_orders**2]
                # scattering = torch.cat([t_scattering, lam], -1)
                yield layer, scattering

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, self.latent, 1, 1)
        z1 = torch.randn(1, self.latent, 1, 1)
        z2 = torch.randn(1, self.latent, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
    
    def log(self, step, data):
        assert isinstance(data, dict), "data must be a dictionary!"
        if self.args.log:
            wandb.log(data, step=step)
        
