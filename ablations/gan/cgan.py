# Modified based on https://github.com/znxlwm/pytorch-generative-model-collections

import gan_utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from torchvision.utils import save_image
from diffusion.sample import evaluate_sampled_results
from utils import utils
from data.dataset import H_MAX, PER
from evaluation.diffraction_measurement import torcwa_simulation
from evaluation.metrics import relative_error
import wandb


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        gan_utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        gan_utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class CGAN(object):
    def __init__(self, args):
        # parameters
        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = args.z_dim
        self.class_num = args.class_num
        self.sample_num = 100

        # load dataset
        self.train_loader = dataloader(self.dataset, self.input_size, self.batch_size, split='train', size_limit=args.size_limit)
        self.test_loader = dataloader(self.dataset, self.input_size, self.sample_num, split='test', size_limit=args.size_limit)
        data = self.train_loader.__iter__().__next__()['layer' if args.dataset == 'MetaLensDataset' else 0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size, class_num=self.class_num)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, class_num=self.class_num)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        gan_utils.print_network(self.G)
        gan_utils.print_network(self.D)
        print('-----------------------------------------------')

        self.sample_z_ = torch.randn(self.sample_num, self.z_dim).to(torch.float32)
        self.fixed_sample = self.test_loader.__iter__().__next__()
        self.sample_y_ = self.fixed_sample['scattering' if args.dataset == 'MetaLensDataset' else 0].to(torch.float32)
        if args.dataset == 'MetaLensDataset':
            self.sample_h_original_ = self.fixed_sample['h_original']

        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    @torch.no_grad()
    def sample(self, fix=True):
        self.G.eval()

        if fix:
            """ fixed noise """
            sample_z_, sample_y_, sample_h_original_ = self.sample_z_, self.sample_y_, self.sample_h_original_
        else:
            """ random noise """
            sample_ = self.test_loader.__iter__().__next__()
            sample_y_ = sample_['scattering'].to(torch.float32)
            sample_h_original_ = sample_['h_original'].to(torch.float32)
            sample_z_ = torch.rand((self.sample_num, self.z_dim)).to(torch.float32)
            if self.gpu_mode:
                sample_z_, sample_y_, sample_h_original_= sample_z_.cuda(), sample_y_.cuda(), sample_h_original_.cuda()

        samples = self.G(sample_z_, sample_y_)

        return samples, sample_y_, sample_h_original_

    @torch.no_grad()
    def evaluate(self, fix=False, verbose=False, return_scatterings=False):
        samples, conditions, h_orig = self.sample(fix)
        t_errors, r_errors = [], []
        actual_scatterings = []
        for b in range(samples.shape[0]):  # foreach item in the batch

            h = H_MAX * samples[b, 1].mean()
            lam = conditions[b][-1]
            p = PER

            phy_kwargs = dict(periodicity=p, h=h.cpu().numpy(), lam=lam.cpu().numpy(), tet=0.0)

            T_desired, R_desired = utils.reshape_conditions(conditions[b].squeeze(0))
            T_actual, R_actual = torcwa_simulation(phy_kwargs, layer=samples[b, 0])

            if return_scatterings:
                actual_scatterings.append(
                    torch.cat([utils.flatten_conditions(T_actual, R_actual), lam.unsqueeze(0).reshape(1, 1)], dim=1))

            t_error, r_error = relative_error(T_actual, T_desired).item(), relative_error(R_actual, R_desired).item()

            if verbose:
                print(f"\tSample #{b:2d} : T_error = {t_error:.4f}, R_error = {r_error:.4f}"
                           f" (original h = {h_orig[b]:.3f}, predicted h = {h:.3f})"
                           f" (wavelength = {lam:.3f})")

            t_errors.append(t_error)
            r_errors.append(r_error)

        if return_scatterings:
            return t_errors, r_errors, torch.cat(actual_scatterings)
        return t_errors, r_errors

    @torch.no_grad()
    def visualize_results(self, epoch, fix=True):

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        samples, _, _ = self.sample(fix)

        # t_error, r_error = evaluate_sampled_results(samples, logger=None, verbose=False) # TODO: Recode evaluation

        fname = os.path.join(self.result_dir ,self.dataset, self.model_name, self.model_name + '_epoch%03d' % epoch + '.png')
        images = samples.mean(dim=(2, 3))[:, 1].reshape(-1, 1, 1, 1) * samples[:, 0:1, ...]
        save_image(images, fname, nrow=image_frame_dim, pad_value=0.5)


    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = utils.now()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = utils.now()
            nof_batches = len(self.train_loader)
            for iter, sample in enumerate(self.train_loader):
                if iter == self.train_loader.dataset.__len__() // self.batch_size:
                    break

                x_ = sample['layer'].to(torch.float32)
                y_ = sample['scattering'].to(torch.float32)

                if self.gpu_mode:
                    x_, y_ = x_.cuda(), y_.cuda()

                z_ = torch.rand((self.batch_size, self.z_dim), device=x_.device)
                y_vec_ = y_  # torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3).expand(self.batch_size, self.class_num, self.input_size, self.input_size)
                # if self.gpu_mode:
                #     x_, z_, y_vec_, y_fill_ = x_.cuda(), z_.cuda(), y_vec_.cuda(), y_fill_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_, y_fill_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_, y_fill_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_, y_fill_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 1000) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.train_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
            self.save()
            with torch.no_grad():
                t_errors, r_errors = self.evaluate()
                self.visualize_results((epoch + 1))

            logs = {'epoch': epoch,
                    'D_loss': np.mean(self.train_hist['D_loss'][-nof_batches:]),
                    'G_loss': np.mean(self.train_hist['G_loss'][-nof_batches:]),
                    't_error': np.mean(t_errors), 'r_error': np.mean(r_errors)}
            if self.args.log:
                wandb.log(logs)
            self.train_hist['per_epoch_time'].append(utils.now() - epoch_start_time)
            logs['per_epoch_time'] = str(self.train_hist['per_epoch_time'][-1])
            print(utils.print_dict_beautifully(logs))

        self.train_hist['total_time'].append(utils.now() - start_time)
        # print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
        #       self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        gan_utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        gan_utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)