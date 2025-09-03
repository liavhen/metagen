import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')
sys.path.append('..')
from ablations.vae_lib.models import *
from ablations.vae_lib.experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from ablations.vae_lib.dataset import VAEDataset
from pytorch_lightning.strategies import DDPStrategy
from data.data_config import get_data_cfg


torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
data_cfg = get_data_cfg(config['data_params']['data_cfg'])
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, data_cfg,
                          config['exp_params'])

data = VAEDataset(data_path=None, **config["data_params"], pin_memory=True)

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
