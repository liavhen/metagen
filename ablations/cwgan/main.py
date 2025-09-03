from gan_utils.config import parse_args
from gan_utils.data_loader import get_data_loader

from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty import WGAN_GP

import torch
from torch.utils.data import DataLoader
from data.lmdb_dataset import MetaLensDatasetLMDB

import wandb
from utils import utils
from os.path import join
from utils.paths import *


torch.set_float32_matmul_precision('medium')

def main(args):
    model = None
    if args.model == 'GAN':
        model = GAN(args)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-CP':
        model = WGAN_CP(args)
    elif args.model == 'WGAN-GP':
        model = WGAN_GP(data_cfg=args.data_cfg, args=args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)
    
    if args.log:
        wandb.login()
        wandb.init(project='metalens-c-wgan-gp', dir='.', name=model.outdir.split('/')[-1], settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)

    dataset_kwargs = dict( data_cfg=args.data_cfg, augments=True)

    dataset = MetaLensDatasetLMDB(**dataset_kwargs)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True, prefetch_factor=4)


    # Load datasets to train and test loaders
    # train_loader, test_loader = get_data_loader(args)

    # train_loader, test_loader = get_data_loader(args)
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    if args.resume_path is not None:
        model.load_pretrained(args.resume_path, resume=True)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)


    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)
    
    wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
