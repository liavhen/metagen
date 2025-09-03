import sys
sys.path.append('.')
sys.path.append('..')
import torch
import wandb
import argparse
import logging
import numpy as np
from utils import utils
from utils.paths import *
import diffusion
from torch.utils.data import random_split
from edm_utils.torch_utils import misc
from data.lmdb_dataset import MetaLensDatasetLMDB
from diffusion.loss import EDMLoss
from schedulefree import AdamWScheduleFree
from diffusion.diffusion_trainer import DiffusionTrainer
from data.data_config import get_data_cfg


def get_parser():
    parser = argparse.ArgumentParser(description='Training Config')
    parser.add_argument('--name', '-N', type=str, required=True)

    # Training Management
    parser.add_argument('--manual_seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--debug_overfit', action='store_true', default=False)
    parser.add_argument('--eval_every', type=int, default=10, help="evaluate model every x time steps")

    # Hyper-Parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', '-B', type=int, default=256)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=3000000)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--lr_schedule_step", type=int, default=50000, help="update lr every x optimizer steps")
    parser.add_argument('--lr_schedule_gamma', type=float, default=0.5, help="lr decay rate every x optimizer steps")


    # Model
    parser.add_argument('--model', type=str, choices=['EDMPrecond', 'VPPrecond', 'VEPrecond'], default='EDMPrecond')
    parser.add_argument('--model_type', type=str, choices=['SongUNet', 'DhariwalUNet', 'DiTL8', 'DiTB8', 'DiTS8'], default='SongUNet')
    parser.add_argument("--model_channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument('--label_dropout', type=float, default=0.1, help="Label dropout for Classifier-Free Guidance")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout for weights regularization")
    parser.add_argument("--ema_halflife", type=int, default=1000)
    parser.add_argument("--ema_warmup_ratio", type=float, default=0.1)
    parser.add_argument('--cw', action='store_true', default=False, help="ConditionsWhitener knob")

    # Data
    parser.add_argument('--data_cfg', type=str, default='a')
    parser.add_argument('--input_res', type=int, default=None)
    parser.add_argument("--size_limit", type=int, default=None)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument('--sorted_dataset', action='store_true', default=False)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--augment_cyc_shift', action='store_true', default=True)
    parser.add_argument('--augment_rotate', action='store_true', default=False)
    parser.add_argument('--augment_flip', action='store_true', default=False)
    parser.add_argument('--max_masked', type=int, default=None)
    parser.add_argument('--override_wavelengths', nargs='+', default=None)
    parser.add_argument('--override_heights', nargs='+', default=None)

    # Loss
    parser.add_argument('--enable_pnn_loss_at', type=int, default=100000, help="Enable PNN loss at this step")
    parser.add_argument('--pnn_loss_weight', type=float, default=0, help="PNN Loss weights")
    parser.add_argument('--pmean', type=float, default=0, help="Mean of the exp-normal distribution of sigma")
    parser.add_argument('--pstd', type=float, default=1.2, help="Std of the exp-normal distribution of sigma")
    parser.add_argument('--use_sdf', action='store_true', default=False)
    parser.add_argument('--sdf_decay_ratio', type=float, default=40,
                        help="Signed Distance Function ratio decay coefficient "
                             "(lower = slower decay; see usage to understand)")
    parser.add_argument('--weight_by_target_projection', action='store_true', default=False)

    # Logging & Verbosity
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--sample_every', type=int, default=20000, help="sample model every x time steps")
    parser.add_argument('--log_every', type=int, default=2000, help='print logs every x time steps')
    parser.add_argument('--save_every', type=int, default=500000, help='save model and results every x time steps')
    parser.add_argument('--save_best', action='store_true', default=True)

    # Misc
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--device', type=int, default=0)


    return parser


def train():
    parser = get_parser()
    args = parser.parse_args()
    # torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    is_debug_run = 'debug' in args.name
    exp_name = args.name
    run_name = exp_name if is_debug_run else exp_name + f'-{utils.get_timestamp()}'
    args.outdir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(args.outdir, exist_ok=True)

    logging.basicConfig(
        filename=f'{args.outdir}/run.log', filemode='w',
        format='%(asctime)s %(levelname)s --> %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info('=========================================================================================')
    logger.info(f'Running experiment {run_name}, {utils.now()}')
    logger.info('=========================================================================================')
    # logger.info(f"Using {args.device} device")

    if args.log:
        wandb.login()
        wandb.init(project='metalens', dir='.', name=run_name, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)

    if args.manual_seed is not None:
        logger.info(f'Manual Seed: {args.manual_seed}')
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError("Input cuda device is not available")

    args.data_cfg = get_data_cfg(args.data_cfg)
    
    if args.input_res is not None:
        args.data_cfg.resolution = args.input_res

    sdf = utils.SDF(sdf_decay=args.sdf_decay_ratio/args.data_cfg.resolution) if args.use_sdf else None

    label_dim = utils.get_label_dim(args.data_cfg)
    model = diffusion.get_model(data_cfg=args.data_cfg, parallel=True, label_dim=label_dim, img_channels=2,
    # model = diffusion.get_model(data_cfg=args.data_cfg, parallel=True, img_channels=2,
                            model_type=args.model_type, model_channels=args.model_channels, num_blocks=args.num_blocks,
                            label_dropout=args.label_dropout,
                            random_init=True).to(device)

    logger.info(f"Identified {torch.cuda.device_count()} GPUs to train on...")
    if args.batch_size % torch.cuda.device_count() != 0:
        new_batch_size = args.batch_size - args.batch_size % torch.cuda.device_count()
        logger.info(f"Batch size ({args.batch_size}) is indivisible by {torch.cuda.device_count()}, updating the batch size to {new_batch_size}...")
        args.batch_size = new_batch_size

    logger.info("\n========================= Arguments ========================")
    for arg in vars(args):
        logger.info(f"\t{arg:<20}: {getattr(args, arg)}")

    if args.model_type in ["DhariwalUNet", "SongUNet"]:
        with torch.no_grad():
            images = torch.zeros([args.batch_size, model.module.img_channels, model.module.img_resolution, model.module.img_resolution], device=device)
            sigma = torch.ones([args.batch_size], device=device)
            labels = torch.zeros([args.batch_size, model.module.label_dim], device=device)
            logger.info("\n========================= Model Summary ========================")
            misc.print_module_summary(model.module, [images, sigma, labels], logger=logger, max_nesting=2)
            logger.info("=================================================================")

    print(f'Learnable parameters: {utils.get_nof_params(model)}')

    # loss_fn = EDMLoss(P_mean=args.pmean, P_std=args.pstd)
    # optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = None

    dataset_kwargs = dict(
        data_cfg=args.data_cfg,
        override_wavelengths=args.override_wavelengths,
        override_heights=args.override_heights,
        initial_scale=args.input_res,
        augments=args.augment_cyc_shift,
        max_masked=args.max_masked,
        size_limit=args.size_limit,
    )


    dataset = MetaLensDatasetLMDB(**dataset_kwargs)
    logger.info(f"{dataset}") # MetaLensDatasetLMDB has a __repr__ method

    pca, target = None, None

    loss_fn = EDMLoss(P_mean=args.pmean, P_std=args.pstd, pca=pca, target=target)
    optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = None

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    trainer = DiffusionTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        sdf=sdf,
        datasets=random_split(dataset, [train_size, test_size]),
        args=args,
        logger=logger
    )

    assert not (args.pretrained and args.resume), "Either 'pretrained' or 'resume' arguments can be provided!"
    if args.pretrained is not None:
        trainer.load_pretrained_model(args.pretrained)
    elif args.resume is not None:
        trainer.resume_training_state(args.resume)
    else:
        logger.info("\nStarts diffusion from scratch, with random initialization")

    logger.info("\n====================================== Starts Training ======================================")
    trainer.train()
    logger.info("\n======================================  Training End.  ======================================")

    wandb.finish()

def main():
    train()


if __name__ == "__main__":
    main()
