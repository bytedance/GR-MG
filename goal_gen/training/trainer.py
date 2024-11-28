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
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from utils.ema import requires_grad,update_ema
from utils.dist_train import get_rank
from model.model import IP2P
class Goalgen_Trainer(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self._main_rank_print('--------------- model configs ---------------')
        self._main_rank_print(configs)
        self.configs = configs
        self._initialize()
        self.save_hyperparameters()     
        self.dataset_name = "calvin"
  
    @staticmethod
    def _main_rank_print(*args, **kwargs):
        if get_rank() == 0:
            print(*args, **kwargs)

    @property
    def num_gpus(self):
        return self.trainer.num_devices * self.trainer.num_nodes

    def _initialize(self):

        self.model = IP2P(
            pretrained_model_dir=self.configs['pretrained_model_dir'], # load the pretrained instructpix2pix weight
            device=self.configs['device'],
            seed=self.configs['seed'],
            conditioning_dropout_prob=self.configs['conditioning_dropout_prob'],
            gradient_checkpointing=self.configs['gradient_checkpointing']
        ) 

        self.use_ema=self.configs["use_ema"]
        if self.use_ema:
            self.ema_model = IP2P(
            pretrained_model_dir=self.configs['pretrained_model_dir'],
            device=self.configs['device'],
            seed=self.configs['seed'],
            conditioning_dropout_prob=self.configs['conditioning_dropout_prob'],
            gradient_checkpointing=self.configs['gradient_checkpointing']
            )
            requires_grad(self.ema_model, False) # ema model will not be trained. It will only be updated.
            self.ema_model.eval()
            

    @classmethod
    def from_checkpoint(cls, ckpt_dir=None, configs=None):
        if ckpt_dir is None:
            assert configs is not None, "ckpt_dir and configs are both None for initialization."
            return cls(configs)

    def configure_optimizers(self):
        lr = self.configs['learning_rate']
        eff_bsz = self.configs['batch_size'] * self.num_gpus
        self._main_rank_print('-' * 40)
        self._main_rank_print(f"learning rate: {lr}, effective batch size: {eff_bsz}")

        optimizer_params = [
                {'params': self.model.unet.parameters(), 'lr': lr},
            ] # only unet will be trained 
 
        optimizer = torch.optim.AdamW(
            optimizer_params,
            betas=(self.configs['adam_beta1'], self.configs['adam_beta2']),
            weight_decay=self.configs['adam_weight_decay'],
            eps=self.configs['adam_epsilon']
        )

        return {
            'optimizer': optimizer
            }

    def _log_output(self, output, phase, dataset=None, **kwargs):
        for k, v in output.items():
            log_name = f"{phase}_{k}"
            if dataset is not None:
                log_name = f"{dataset}_{log_name}"
            self.log(log_name, v, prog_bar=True, **kwargs)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            if self.use_ema:
                prediction, target = self.ema_model.forward(batch)
            else:
                prediction, target = self.model.forward(batch)
            loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
            output = {'loss': loss}

            self._log_output(output, phase="val", sync_dist=True, 
                             on_epoch=True, on_step=False, dataset=self.dataset_name)

    def training_step(self, batch, batch_idx):
        prediction, target = self.model.forward(batch)
        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
        output = {'loss': loss}
        self._log_output(output, phase="train", on_epoch=False, on_step=True,dataset=self.dataset_name)
        if self.configs['use_ema']:
            update_ema(self.ema_model, self.model, decay=0.999)
        return output['loss']
        
    
    def on_save_checkpoint(self, checkpoint):
        if not self.use_ema:
            checkpoint['state_dict'] = {'unet': self.model.unet.state_dict()}
        else:
            checkpoint['state_dict'] = {'unet_ema': self.ema_model.unet.state_dict(),'unet': self.model.unet.state_dict()}
    def on_load_checkpoint(self, checkpoint):
        if not self.use_ema:
            self.model.unet.load_state_dict(checkpoint['state_dict']['unet'])
        else:
            self.model.unet.load_state_dict(checkpoint['state_dict']['unet_ema'])
            self.ema_model.unet.load_state_dict(checkpoint['state_dict']['unet_ema'])