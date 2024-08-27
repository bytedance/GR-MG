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
import torch
import torch.nn.functional as F
from functools import partial
import math
import lightning.pytorch as pl
from utils.dist_train import get_rank
import model.vision_transformer as vits
import json
from model.model import GR_MG
import clip
def adjust_learning_rate(iter, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_iters = configs['warmup_iters']
    total_iters = configs['iters']
    min_lr_scale = configs['min_lr_scale']

    if iter < configs['warmup_iters']:
        lr_scaler = 1.0 * iter / warmup_iters
    else:
        lr_scaler = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * \
            (1.0 + math.cos(math.pi * (iter - warmup_iters) / (total_iters - warmup_iters)))
    return lr_scaler

def compute_kl_divergence( mu, logvar):
    if torch.isnan(mu).any() or torch.isnan(logvar).any():
        raise ValueError("Input contains NaN values")
    if torch.isinf(mu).any() or torch.isinf(logvar).any():
        raise ValueError("Input contains infinite values")
    type=mu.dtype
    latent_dim = mu.shape[-1]
    # 将 mu 和 logvar 转换为 float32
    mu = mu.float().view(-1, latent_dim)
    logvar = logvar.float().view(-1, latent_dim)
    
    klds = -0.5 * (1 + logvar- mu.pow(2) - logvar.exp())
    klds = klds.sum(1).mean(0, True).squeeze()
    
    # transform back to bf16
    return klds.to(type)


class Policy_Trainer(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self._main_rank_print('--------------- model configs ---------------')
        self._main_rank_print(configs)
        self.configs = configs
        self._initialize()
        self.save_hyperparameters()       
        self.val_set_names = "calvin"

    @staticmethod
    def _main_rank_print(*args, **kwargs):
        if get_rank() == 0:
            print(*args, **kwargs)

    @property
    def num_gpus(self):
        return self.trainer.num_devices * self.trainer.num_nodes

    def _initialize(self):
        training_target = []
        if self.configs["trainer"]['act_pred']:
            training_target.append('act_pred')
        if self.configs["trainer"]['fwd_pred']: # predict future static image
            training_target.append('fwd_pred')
        if self.configs["trainer"]['fwd_pred_hand']: # predict future hand image
            training_target.append('fwd_pred_hand')
        if self.configs["trainer"]['progress_pred']: # predict progress information
            training_target.append('progress_pred')

        # mae model
        model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        model_mae.to(self.device)
        mae_ckpt = '/PATH_TO/resources/MAE/mae_pretrain_vit_base.pth'
        checkpoint = torch.load(mae_ckpt, map_location='cpu')
        model_mae.load_state_dict(checkpoint['model'], strict=True)
        # freeze mae
        for name, p in model_mae.named_parameters():
            p.requires_grad = False
        #clip model
        clip_name = "ViT-B/32"
        clip_model, clip_preprocess = clip.load(clip_name)
        # freeze clip
        for _, param in clip_model.named_parameters():
            param.requires_grad = False
        
        # resampler parameters
        resampler_params = dict()
        resampler_params['depth'] = self.configs["policy"]['resampler_params']["depth"]
        resampler_params['dim_head'] = self.configs["policy"]['resampler_params']['dim_head']
        resampler_params['heads'] = self.configs["policy"]['resampler_params']['heads']
        resampler_params['num_latents'] = self.configs["policy"]['resampler_params']['num_latents']
        resampler_params['num_media_embeds'] = self.configs["policy"]['resampler_params']['num_media_embeds']

        # main model
        self.model = GR_MG(
        state_dim=self.configs["input"]['state_dim'],
        act_dim=self.configs["input"]['act_dim'],
        act_len=self.configs["policy"]['act_len'],
        act_latent_dim=self.configs["policy"]['act_latent_dim'],
        act_encoder_dim=self.configs["policy"]['act_encoder_dim'],
        act_decoder_dim=self.configs["policy"]['act_decoder_dim'],
        progress_decoder_dim=self.configs["policy"]["progress_decoder_dim"],
        hidden_size=self.configs["policy"]['embed_dim'],
        model_mae=model_mae,
        clip_model=clip_model,
        img_feat_dim=self.configs["policy"]["img_feat_dim"],
        lang_feat_dim = self.configs["policy"]["lang_feat_dim"],
        patch_feat_dim=self.configs["policy"]["patch_feat_dim"],
        resampler_params=resampler_params,
        max_length=self.configs["policy"]['seq_len'],
        training_target=training_target,
        without_norm_pix_loss=self.configs["trainer"]['without_norm_pix_loss'],
        use_hand_rgb=self.configs["input"]['use_hand_rgb'],
        use_state=self.configs["input"]['use_state'],
        use_resampler=self.configs["policy"]['use_resampler'],
        n_layer=self.configs["policy"]['n_layer'],
        n_head=self.configs["policy"]['n_head'],
        n_inner=4*self.configs["policy"]['embed_dim'],
        activation_function=self.configs["policy"]['activation_function'],
        n_positions=1024,
        resid_pdrop=self.configs["policy"]['dropout'],
        attn_pdrop=self.configs["policy"]['dropout'])

    
        # if finetune, we need to load the pretrained model
        if self.configs["trainer"]["finetune"]:
            if self.configs["trainer"]["use_pretrain"]:
                trainer_config=self.configs["trainer"]
                self._main_rank_print(f"Loading pretrained model from: {trainer_config['pretrained_model_path']}")
                checkpoint = torch.load(self.configs["trainer"]['pretrained_model_path'], map_location='cpu')
                state_dict = dict()
                # Exclude action and state related weights
                for key, value in checkpoint['state_dict'].items():
                    if key[:6]=="model.":
                        key = key[6:] # remove "model." from pl checkpoint
                    state_dict[key] = value
                del checkpoint
                msg = self.model.load_state_dict(state_dict, strict=False)
                self._main_rank_print(msg)
                del state_dict

        # save config
        if get_rank() == 0:
            with open(os.path.join(self.configs['ckpt_dir'], 'hyperparameters.json'), 'w') as f:
                json.dump(self.configs, f)


        # these variables are used to indicate what information will be used or predicted
        self.act_pred = self.model.act_pred
        self.fwd_pred = self.model.fwd_pred
        self.fwd_pred_hand = self.model.fwd_pred_hand
        self.use_state = self.model.use_state
        self.use_hand_rgb = self.model.use_hand_rgb
        self.progress_pred=self.model.progress_pred

        # loss
        self.kl_loss_ratio =self.configs["trainer"]["kl_loss_ratio"]
        self.gripper_loss_ratio =self.configs["trainer"]["gripper_loss_ratio"]
        self.fwd_loss_ratio =self.configs["trainer"]["fwd_loss_ratio"]
        self.progress_loss_ratio=self.configs["trainer"]["progress_loss_ratio"]
            


    def configure_optimizers(self):
        lr = self.configs["trainer"]['learning_rate']
        eff_bsz = self.configs["trainer"]['batch_size'] * self.num_gpus
        self._main_rank_print('-' * 40)
        self._main_rank_print("LR SCHEDULER CONFIGS:")
        self._main_rank_print(f"learning rate: {lr}, effective batch size: {eff_bsz}")
       
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(self.configs["trainer"]['betas_1'], self.configs["trainer"]['betas_2']),
            weight_decay=self.configs["trainer"]['weight_decay']
        )
        assert self.trainer.max_epochs is not None
        num_training_batches = self.trainer.estimated_stepping_batches
        iter_per_epoch = num_training_batches / self.trainer.max_epochs
        self.configs["trainer"]['warmup_steps']=self.configs["trainer"]['warmup_epochs'] * iter_per_epoch
        lr_scheduler_configs = {
            'warmup_iters': self.configs["trainer"]['warmup_steps'],
            'iters': self.trainer.max_epochs * iter_per_epoch,
            'min_lr_scale': self.configs["trainer"]['min_learning_rate_scale']
        }
        lr_lambda = partial(adjust_learning_rate, configs=lr_scheduler_configs)
        self._main_rank_print(lr_scheduler_configs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }


    def _log_output(self, output, phase, dataset=None, **kwargs):
        for k, v in output.items():
            log_name = f"{phase}_{k}"
            if dataset is not None:
                log_name = f"{dataset}_{log_name}"
            self.log(log_name, v, prog_bar=True, **kwargs)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            goal_rgb = batch['goal_rgb']
            rgb = batch['rgb']
            hand_rgb = batch['hand_rgb']
            state = batch['rel_state']
            action = batch['action']
            action_mask = batch['action_mask']
            attention_mask = batch['attention_mask']
            text=batch["text"]
            progress=batch["progress"]

            # Split arm and gripper action
            arm_action = action[:, :, :, :6]  # (b, l, act_len, act_dim - 1)
            gripper_action = action[:, :, :, 6]  # (b, l, act_len)
            arm_action_target = torch.clone(arm_action)  # (b, l, act_len, act_dim - 1)
            gripper_action_target = torch.clone(gripper_action)  # (b, l, act_len)

            # Split arm and gripper state
            arm_state = state[:, :, :6]  # (b, l, state_dim - 1)
            gripper_state = state[:, :, 6].long()  # (b, l)
            gripper_state = F.one_hot(gripper_state, num_classes=2).type_as(arm_state)  # (b, l, 2)

            seq_len = arm_action.size(1)
            act_len = arm_action.size(2)

            input_dict = {
                'goal_rgb': goal_rgb,
                'rgb': rgb,
                'hand_rgb': hand_rgb,
                'arm_action': arm_action,
                'gripper_action': gripper_action,
                'arm_state': arm_state,
                'gripper_state': gripper_state,
                'attention_mask': attention_mask, #（b,l)
                'text':text,
                'progress':progress
            }
        
        
            prediction = self.model(input_dict, is_training=False)


            obs_preds = prediction['obs_preds']
            obs_target = prediction['obs_target']
            arm_action_preds = prediction['arm_action_preds']  # (b, l, act_len, act_dim - 1)
            gripper_action_preds = prediction['gripper_action_preds']  # (b, l, act_len, 1)
            obs_hand_preds = prediction['obs_hand_preds']
            obs_hand_target = prediction['obs_hand_target']
            action_mu_preds = prediction['action_mu_preds']  # (b * l, act_latent_dim)
            action_logvar_preds = prediction['action_logvar_preds']  # (b * l, act_latent_dim)
            progress_preds=prediction["progress_preds"]
            progress_targets=prediction["progress_targets"]


            loss_act = 0
            loss_arm_act = 0
            loss_gripper_act = 0
            loss_kl_act = 0
            acc_gripper_act = 0
            loss_obs = 0
            loss_hand_obs = 0
            gripper_cnt = 0
            loss_progress=0

            # action prediction
            act_dim = self.model.act_dim
            if self.act_pred:
                # kl loss
                loss_kl_act = compute_kl_divergence(action_mu_preds, action_logvar_preds)
  

                # action smooth l1 loss
                arm_action_preds = arm_action_preds.view(-1, act_len, act_dim-1)[attention_mask.flatten() > 0] # b,len,act_len,6 -> b*len,act_len,6
                arm_action_target = arm_action_target.view(-1, act_len, act_dim-1)[attention_mask.flatten() > 0] # b,len,act_len,6 -> b*len,act_len,6
                action_mask = action_mask.view(-1, act_len)[attention_mask.flatten() > 0] # b,len,act_len -> b*len,act_len
                arm_action_preds = arm_action_preds.view(-1, act_dim-1)[action_mask.flatten() > 0] # b*len*act_len, 6
                arm_action_target = arm_action_target.view(-1, act_dim-1)[action_mask.flatten() > 0] # b*len*act_len, 6
                loss_arm_act = torch.nn.SmoothL1Loss()(arm_action_preds, arm_action_target)

                # gripper bce loss
                gripper_action_preds = gripper_action_preds.view(-1, act_len)[attention_mask.flatten() > 0] # b,len,act_len -> b*len,act_len
                gripper_action_target = gripper_action_target.view(-1, act_len)[attention_mask.flatten() > 0] # b,len,act_len -> b*len,act_len
                gripper_action_preds = gripper_action_preds.flatten()[action_mask.flatten() > 0] # b*len*act_len
                gripper_action_target = gripper_action_target.flatten()[action_mask.flatten() > 0] # b*len*act_len
                bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
                loss_gripper_act = bce_with_logits_loss(gripper_action_preds, gripper_action_target)
                loss_act = loss_arm_act + loss_gripper_act * self.gripper_loss_ratio + loss_kl_act * self.kl_loss_ratio
                
                # Compute gripper action acc
                gripper_action_preds = torch.nn.Sigmoid()(gripper_action_preds) # Sigmoid function
                gripper_action_preds = (gripper_action_preds > 0.5).float()
                acc_gripper_act = torch.eq(gripper_action_preds, gripper_action_target).sum().float()
                gripper_cnt = gripper_action_preds.shape[0]
                acc_gripper_act /= gripper_cnt

            # forward prediction
            if self.fwd_pred:
                fwd_pred_next_n = self.configs["trainer"]['fwd_pred_next_n']
                obs_preds = obs_preds[:, :seq_len-fwd_pred_next_n, :, :]
                obs_target = obs_target[:, fwd_pred_next_n:, :, :]
                obs_attention_mask = attention_mask[:, fwd_pred_next_n:]
                loss_obs = (obs_preds - obs_target) ** 2
                loss_obs = loss_obs.mean(dim=-1).mean(dim=-1)
                loss_obs = (loss_obs * obs_attention_mask).sum() / obs_attention_mask.sum()
                if self.fwd_pred_hand:
                    obs_hand_preds = obs_hand_preds[:, :seq_len-fwd_pred_next_n, :, :]
                    obs_hand_target = obs_hand_target[:, fwd_pred_next_n:, :, :]
                    loss_hand_obs = (obs_hand_preds - obs_hand_target) ** 2
                    loss_hand_obs = loss_hand_obs.mean(dim=-1).mean(dim=-1)
                    loss_hand_obs = (loss_hand_obs * obs_attention_mask).sum() / obs_attention_mask.sum()
            if self.progress_pred:
                diff = progress_preds - progress_targets
                masked_diff = diff * attention_mask
                squared_error = masked_diff ** 2
                loss_progress = squared_error.sum() / attention_mask.sum()

            # compute loss
            loss = torch.tensor(0.0).to(self.device)
            if self.act_pred:
                loss += loss_act
            if self.fwd_pred:
                loss += self.fwd_loss_ratio * loss_obs
                if self.fwd_pred_hand:
                    loss += self.fwd_loss_ratio * loss_hand_obs
            if self.progress_pred:
                loss+=loss_progress*self.progress_loss_ratio
            output = {
                'loss': loss,
                'loss_act': loss_act,
                'loss_arm_act': loss_arm_act,
                'loss_gripper_act': loss_gripper_act,
                'loss_kl_act': loss_kl_act,
                'acc_gripper_act': acc_gripper_act,
                'loss_obs': loss_obs,
                'loss_hand_obs': loss_hand_obs,
                'loss_progress':loss_progress
            }
            self._log_output(output, phase="val", on_epoch=True, on_step=False)
            return output['loss']

    def training_step(self, batch, batch_idx):
        goal_rgb = batch['goal_rgb']
        rgb = batch['rgb']
        hand_rgb = batch['hand_rgb']
        state = batch['rel_state']
        action = batch['action']
        action_mask = batch['action_mask']
        attention_mask = batch['attention_mask']
        text=batch["text"]
        progress=batch["progress"]
        # Split arm and gripper action
        arm_action = action[:, :, :, :6]  # (b, l, act_len, act_dim - 1)
        gripper_action = action[:, :, :, 6]  # (b, l, act_len)
        arm_action_target = torch.clone(arm_action)  # (b, l, act_len, act_dim - 1)
        gripper_action_target = torch.clone(gripper_action)  # (b, l, act_len)

        # Split arm and gripper state
        arm_state = state[:, :, :6]  # (b, l, state_dim - 1)
        gripper_state = state[:, :, 6].long()  # (b, l)
        gripper_state = F.one_hot(gripper_state, num_classes=2).type_as(arm_state)  # (b, l, 2)

        seq_len = arm_action.size(1)
        act_len = arm_action.size(2)

        input_dict = {
            'goal_rgb': goal_rgb,
            'rgb': rgb,
            'hand_rgb': hand_rgb,
            'arm_action': arm_action,
            'gripper_action': gripper_action,
            'arm_state': arm_state,
            'gripper_state': gripper_state,
            'attention_mask': attention_mask,
            'text':text,
            'progress':progress
        }
    
    
        prediction = self.model(input_dict, is_training=True)


        obs_preds = prediction['obs_preds']
        obs_target = prediction['obs_target']
        arm_action_preds = prediction['arm_action_preds']  # (b, l, act_len, act_dim - 1)
        gripper_action_preds = prediction['gripper_action_preds']  # (b, l, act_len, 1)
        obs_hand_preds = prediction['obs_hand_preds']
        obs_hand_target = prediction['obs_hand_target']
        action_mu_preds = prediction['action_mu_preds']  # (b * l, act_latent_dim)
        action_logvar_preds = prediction['action_logvar_preds']  # (b * l, act_latent_dim)
        progress_preds=prediction["progress_preds"]
        progress_targets=prediction["progress_targets"]



        loss_act = 0
        loss_arm_act = 0
        loss_gripper_act = 0
        loss_kl_act = 0
        acc_gripper_act = 0
        loss_obs = 0
        loss_hand_obs = 0
        gripper_cnt = 0
        loss_progress= 0
        # action prediction
        act_dim = self.model.act_dim
        if self.act_pred:
            # kl loss
         
            loss_kl_act = compute_kl_divergence(action_mu_preds, action_logvar_preds)


            # action smooth l1 loss
            arm_action_preds = arm_action_preds.view(-1, act_len, act_dim-1)[attention_mask.flatten() > 0] # b,len,act_len,6 -> b*len,act_len,6
            arm_action_target = arm_action_target.view(-1, act_len, act_dim-1)[attention_mask.flatten() > 0] # b,len,act_len,6 -> b*len,act_len,6
            action_mask = action_mask.view(-1, act_len)[attention_mask.flatten() > 0] # b,len,act_len -> b*len,act_len
            arm_action_preds = arm_action_preds.view(-1, act_dim-1)[action_mask.flatten() > 0] # b*len*act_len, 6
            arm_action_target = arm_action_target.view(-1, act_dim-1)[action_mask.flatten() > 0] # b*len*act_len, 6
            loss_arm_act = torch.nn.SmoothL1Loss()(arm_action_preds, arm_action_target)

            # gripper bce loss
            gripper_action_preds = gripper_action_preds.view(-1, act_len)[attention_mask.flatten() > 0] # b,len,act_len -> b*len,act_len
            gripper_action_target = gripper_action_target.view(-1, act_len)[attention_mask.flatten() > 0] # b,len,act_len -> b*len,act_len
            gripper_action_preds = gripper_action_preds.flatten()[action_mask.flatten() > 0] # b*len*act_len
            gripper_action_target = gripper_action_target.flatten()[action_mask.flatten() > 0] # b*len*act_len
            bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
            loss_gripper_act = bce_with_logits_loss(gripper_action_preds, gripper_action_target)
            loss_act = loss_arm_act + loss_gripper_act * self.gripper_loss_ratio + loss_kl_act * self.kl_loss_ratio
            
            # Compute gripper action acc
            gripper_action_preds = torch.nn.Sigmoid()(gripper_action_preds) 
            gripper_action_preds = (gripper_action_preds > 0.5).float()
            acc_gripper_act = torch.eq(gripper_action_preds, gripper_action_target).sum().float()
            gripper_cnt = gripper_action_preds.shape[0]
            acc_gripper_act /= gripper_cnt

        # predict future image
        if self.fwd_pred:
            fwd_pred_next_n = self.configs["trainer"]['fwd_pred_next_n']
            obs_preds = obs_preds[:, :seq_len-fwd_pred_next_n, :, :]
            obs_target = obs_target[:, fwd_pred_next_n:, :, :]
            obs_attention_mask = attention_mask[:, fwd_pred_next_n:]
            loss_obs = (obs_preds - obs_target) ** 2
            loss_obs = loss_obs.mean(dim=-1).mean(dim=-1)
            loss_obs = (loss_obs * obs_attention_mask).sum() / obs_attention_mask.sum()
            if self.fwd_pred_hand:
                obs_hand_preds = obs_hand_preds[:, :seq_len-fwd_pred_next_n, :, :]
                obs_hand_target = obs_hand_target[:, fwd_pred_next_n:, :, :]
                loss_hand_obs = (obs_hand_preds - obs_hand_target) ** 2
                loss_hand_obs = loss_hand_obs.mean(dim=-1).mean(dim=-1)
                loss_hand_obs = (loss_hand_obs * obs_attention_mask).sum() / obs_attention_mask.sum()
        
        if self.progress_pred:
            diff = progress_preds - progress_targets
            masked_diff = diff * attention_mask
            squared_error = masked_diff ** 2
            loss_progress = squared_error.sum() / attention_mask.sum()

        # compute loss
        loss = torch.tensor(0.0).to(self.device)
        if self.act_pred:
            loss += loss_act
        if self.fwd_pred:
            loss += self.fwd_loss_ratio * loss_obs
            if self.fwd_pred_hand:
                loss += self.fwd_loss_ratio * loss_hand_obs
        if self.progress_pred:
            loss+=loss_progress*self.progress_loss_ratio
        output = {
            'loss': loss,
            'loss_act': loss_act,
            'loss_arm_act': loss_arm_act,
            'loss_gripper_act': loss_gripper_act,
            'loss_kl_act': loss_kl_act,
            'acc_gripper_act': acc_gripper_act,
            'loss_obs': loss_obs,
            'loss_hand_obs': loss_hand_obs,
            'loss_progress':loss_progress
        }
        self._log_output(output, phase="train", on_epoch=False, on_step=True)
        return output['loss']
        
    

