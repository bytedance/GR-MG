# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json
from pathlib import Path
import importlib
import copy
import numpy as np
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything
import torch
import random
from utils.utils import SetupCallback
import datetime
from data.calvindataset import CalvinDataset_Goalgen
from goal_gen.training.trainer import Goalgen_Trainer
from torch.utils.data import DataLoader
def get_date_str():
    return str(datetime.date.today())


def init_setup_callback(config):
    return SetupCallback(
        now=str(datetime.datetime.now()).replace(' ', '_'),
        logdir=config['log_dir'],
        ckptdir=config['ckpt_dir'],
    )

def init_trainer_config(configs):
    trainer_config = copy.deepcopy(configs['trainer'])
    trainer_config['devices'] = configs.get('gpus', 'auto')
    trainer_config['num_nodes'] = configs.get('num_nodes', 1)
    if 'strategy' not in trainer_config or trainer_config['strategy'] == 'ddp':
        trainer_config['strategy'] = DDPStrategy(find_unused_parameters=False)
    exp_name = configs['exp_name']

    # init loggers
    loggers = None
    log_dir = os.path.join(get_date_str(), exp_name)
    log_dir = os.path.join(configs['log_root'], log_dir)
    configs['log_dir'] = log_dir
    Path(configs['log_dir']).mkdir(parents=True, exist_ok=True)
    loggers = []
    loggers.append(TensorBoardLogger(log_dir, name=exp_name)) # you can also add other loggers 
    trainer_config['logger'] = loggers
    ckpt_dir = os.path.join(get_date_str(), exp_name)
    ckpt_dir = os.path.join(configs['ckpt_root'], ckpt_dir)
    configs['ckpt_dir'] = ckpt_dir
    Path(configs['ckpt_dir']).mkdir(parents=True, exist_ok=True)
    trainer_config['callbacks'] = [
        init_setup_callback(configs),
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1,every_n_epochs=1)
    ]
    return trainer_config


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)


def experiment(variant):
    set_seed(variant['seed'])
    trainer_config = init_trainer_config(variant)

    trainer = Trainer(**trainer_config)
    variant['gpus'] = trainer.num_devices

    model = Goalgen_Trainer(variant)

    # dataset
    train_data= CalvinDataset_Goalgen(
                data_dir="/PATH_TO_CALVIN/calvin/task_ABC_D/",
                resolution=256,
                resolution_before_crop=288,
                center_crop=False,
                forward_n_min_max=[20, 22],
                use_full=True,
                is_training=True,
                color_aug=True)
    val_data= CalvinDataset_Goalgen(
                data_dir="/PATH_TO_CALVIN/calvin/task_ABC_D/",
                resolution=256,
                resolution_before_crop=288,
                center_crop=False,
                forward_n_min_max=[20, 22],
                use_full = True,
                is_training=False,
                color_aug=False)
    train_dataloader= DataLoader(train_data, 
        batch_size=variant["batch_size"],
        num_workers=variant["num_workers"])
    val_dataloader= DataLoader(val_data, 
        batch_size=variant["batch_size"],
        num_workers=variant["num_workers"])

    _kwargs = {
        'model': model,
        'train_dataloaders':train_dataloader,
        'val_dataloaders':val_dataloader,
        'ckpt_path': variant['resume']
    }
    if _kwargs['ckpt_path'] is not None:
        print(f"Resuming from {variant['resume']}...")
    trainer.fit(**_kwargs)

def deep_update(d1, d2):
    # use d2 to update d1
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1

def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get('parent', None):
        deep_update(config, load_config(_config['parent']))
    deep_update(config, _config)
    return config

def update_configs(configs, args):
    for (k, v) in args.items():
        if k not in configs:
            print(f"{k} not in config. The value is {v}.")
            configs[k] = v
        if isinstance(v, dict):
            for (sub_k, sub_v) in v.items():
                assert sub_k in configs[k], f"{sub_k} - {configs[k]}"
                if sub_v != None:
                    configs[k][sub_k] = sub_v
        else:
            if v != None:
                configs[k] = v
    return configs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default="")
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_root', default=None, type=str)
    parser.add_argument('--ckpt_root', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)

    # Training
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--min_lr_scale', default=None, type=float)
    parser.add_argument('--warmup_steps', default=None, type=int)
    parser.add_argument('--adam_weight_decay', default=None, type=float)
    parser.add_argument('--adam_beta1', default=None, type=float)
    parser.add_argument('--adam_beta2', default=None, type=float)
    parser.add_argument('--adam_epsilon', default=None, type=float)

    # Diffusion
    parser.add_argument("--conditioning_dropout_prob", default=None, type=float)
    global_names = set(vars(parser.parse_known_args()[0]).keys())
    
    # Trainer
    trainer_parser = parser.add_argument_group('trainer')
    trainer_parser.add_argument('--strategy', default=None, type=str)
    trainer_parser.add_argument('--precision', default=None, type=str)
    trainer_parser.add_argument('--gradient_clip_val', default=None, type=float)
    trainer_parser.add_argument('--max_epochs', default=None, type=int)
    trainer_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names

    args = {}
    trainer_args = {}
    temp_args = vars(parser.parse_args())
    for (k, v) in temp_args.items():
        if k in global_names:
            args[k] = v
        elif k in trainer_names:
            trainer_args[k] = v

    args['trainer'] = trainer_args
    
    return args

if __name__ == '__main__':
    args=parse_args()
    configs = load_config(args.pop('config'))
    configs = update_configs(configs, args)
    os.system(f"sudo chmod 777 -R {configs['ckpt_root']}")
    os.system(f"sudo chmod 777 -R {configs['log_root']}")
    experiment(variant=configs)
