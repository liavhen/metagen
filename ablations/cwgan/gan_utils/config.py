import argparse
import os
import sys
sys.path.append('.')
sys.path.append('..')
from data.data_config import get_data_cfg
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

    parser.add_argument('--model', type=str, default='WGAN-GP', choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP'])

    parser.add_argument('--data_cfg', type=str, default='a1', help='data config')
    parser.add_argument('--latent_dim', type=int, default=100, help='latent vectors (z) dimension')
    parser.add_argument('--img_channels', type=int, default=2, help='img channels dimension')

    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--resume_path', type=str, default=None, help='path to state dict from training resume')
    # parser.add_argument('--dataroot', required=True, help='path to dataset')
    # parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'], help='The name of dataset')
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')

    parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
    parser.add_argument('--generator_iters', type=int, default=500000, help='The number of iterations for generator in WGAN model.')
    parser.add_argument('--critic_iters', type=int, default=1, help='The number of critic iterations for every iteration of the generator in WGAN model.')

    parser.add_argument('--log', action='store_true', default=False, help='Whether to log in wandb')
    parser.add_argument('--log_every', type=int, default=1000, help='stats logging frequency')
    parser.add_argument('--eval_every', type=int, default=10000, help='evaluation frequency')
    parser.add_argument('--image_every', type=int, default=25000, help='image and model logging frequency')

    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    args.data_cfg = get_data_cfg(args.data_cfg)
    args.label_dim = args.data_cfg.info_t_orders ** 2 + 1  # 1 for wavelength

    # if args.dataset == 'cifar' or args.dataset == 'stl10':
    #     args.channels = 3
    # else:
    #     args.channels = 1
    args.cuda = args.cuda and torch.cuda.is_available()
    return args
