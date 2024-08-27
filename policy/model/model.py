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
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import transformers
from flamingo_pytorch import PerceiverResampler
import clip
from policy.model.gpt2 import GPT2Model
from policy.utils.dist_train import print as dis_print
import os
from policy.model.vision_transformer import Block,get_2d_sincos_pos_embed

def reparameterize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class GR_MG(nn.Module):
    def __init__(
            self,
            state_dim,
            act_dim,
            act_len,
            act_latent_dim,
            act_encoder_dim,
            act_decoder_dim,
            progress_decoder_dim,
            hidden_size,
            model_mae,
            clip_model,
            img_feat_dim,
            lang_feat_dim,
            patch_feat_dim,
            resampler_params,
            max_length=None,
            training_target=['act_pred'],
            without_norm_pix_loss=False,
            use_hand_rgb=False,
            use_state=False,
            use_resampler=True,
            **kwargs):
        super(GR_MG, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        # note: the difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves) 
        # and wte is removed ( to set find_unused_parameters=False)
        self.transformer = GPT2Model(config)
        transformer_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        dis_print(f"Transformer Parameters: {transformer_params / 1000000:.2f}M")

        self.use_resampler = use_resampler
        if self.use_resampler:
            self.n_patch_latents = resampler_params['num_latents']
            self.perceiver_resampler = PerceiverResampler(
                dim=patch_feat_dim,
                depth=resampler_params['depth'],
                dim_head=resampler_params['dim_head'],
                heads=resampler_params['heads'],
                num_latents=self.n_patch_latents,
                num_media_embeds=resampler_params['num_media_embeds'])
        
            resampler_params = sum(p.numel() for p in self.perceiver_resampler.parameters() if p.requires_grad)
            dis_print(f"Perceiver Resampler Parameters: {resampler_params / 1000000:.2f}M")

        self.model_mae = model_mae
        
        self.act_len = act_len
        self.act_latent_dim = act_latent_dim
        self.act_encoder_dim = act_encoder_dim
        self.act_decoder_dim = act_decoder_dim
        self.progress_decoder_dim=progress_decoder_dim

        self.use_hand_rgb = use_hand_rgb
        self.use_state = use_state

        self.text_tokenizer=clip.tokenize
        self.text_encoder=clip_model


        self.lang_feat_dim=lang_feat_dim  # hardcode
        self.img_feat_dim = img_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.n_patches = 49 # TODO: hardcode
        self.patch_size = 16 # TODO: hardcode
        self.image_size = 224 # TODO: hardcode

        self.act_pred = False
        self.fwd_pred = False
        self.fwd_pred_hand = False
        self.progress_pred=False
        if 'act_pred' in training_target:
            self.act_pred = True
        if 'fwd_pred' in training_target:
            self.fwd_pred = True
        if 'fwd_pred_hand' in training_target:
            self.fwd_pred_hand = True
        if 'progress_pred' in training_target:
            self.progress_pred = True
        
        self.without_norm_pix_loss = without_norm_pix_loss
        if self.use_state:
            # state embedding
            self.embed_arm_state = torch.nn.Linear(self.state_dim-1, self.hidden_size)
            self.embed_gripper_state = torch.nn.Linear(2, self.hidden_size) # one-hot gripper state
            self.embed_state = torch.nn.Linear(2*self.hidden_size, self.hidden_size)


        # Embedding function for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, self.hidden_size)
        # relative timestep embedding
        self.embed_timestep = nn.Embedding(self.max_length, self.hidden_size)

        # image token embedding
        if self.use_hand_rgb:
            self.embed_hand_img = torch.nn.Linear(self.img_feat_dim, self.hidden_size)
        self.embed_img = torch.nn.Linear(self.img_feat_dim, self.hidden_size)
        self.embed_goal_image = torch.nn.Linear(self.img_feat_dim, self.hidden_size)

        # patch token embedding
        if self.use_hand_rgb:
            self.embed_hand_patch = torch.nn.Linear(self.patch_feat_dim, self.hidden_size)
        self.embed_patch = torch.nn.Linear(self.patch_feat_dim, self.hidden_size)
        self.embed_goal_patch = torch.nn.Linear(self.patch_feat_dim, self.hidden_size)

        # layer norm
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        if self.act_pred:
             # action query [ACT]
            self.action_queries = nn.Embedding(1, self.hidden_size) # arm + gripper

            # action encoder (embed action trajectory as style vector)
            self.embed_arm_action = torch.nn.Linear(self.act_dim - 1, self.act_encoder_dim)
            self.embed_gripper_action = torch.nn.Embedding(2, self.act_encoder_dim)
            self.embed_action = nn.Linear(2 * self.act_encoder_dim, self.act_encoder_dim)
            self.action_encoder_cls_token = torch.nn.Embedding(1, self.act_encoder_dim)
            action_encoder_depth = 4
            self.encode_action = nn.ModuleList([
                Block(self.act_encoder_dim, 8, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
                for i in range(action_encoder_depth)])
            self.action_encoder_positional_embeddings = nn.Embedding(self.act_len + 1, self.act_encoder_dim)
            self.pred_style_vector = nn.Linear(self.act_encoder_dim, 2 * self.act_latent_dim)
            self.embed_style_vector = nn.Linear(self.act_latent_dim, self.act_decoder_dim)
        
            # action decoder
            self.proj_action_output_embed = nn.Linear(self.hidden_size, self.act_decoder_dim)
            action_decoder_depth = 4
            self.decode_action = nn.ModuleList([
                Block(self.act_decoder_dim, 8, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
                for i in range(action_decoder_depth)])
            self.action_mask_token_embedding = nn.Embedding(1, self.act_decoder_dim)
            self.action_decoder_positional_embeddings = nn.Embedding(self.act_len, self.act_decoder_dim)
            self.pred_arm_act = nn.Linear(self.act_decoder_dim, self.act_dim - 1) # arm action
            self.pred_gripper_act = nn.Linear(self.act_decoder_dim, 1) # gripper action (binary)
            
        # predict future image
        if self.fwd_pred:
            # add observation query for fwd prediction
            self.obs_queries = nn.Embedding(self.n_patch_latents+1, self.hidden_size) # cls+resampler
            if self.use_hand_rgb:
                self.obs_hand_queries = nn.Embedding(self.n_patch_latents+1, self.hidden_size) # cls+resampler
            self.decoder_embed = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_size))
            # fixed sin-cos embedding
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (self.image_size//self.patch_size)**2,
                self.hidden_size), requires_grad=False)  # (1, n_patch, h)

            decoder_depth = 2 # hardcode
            self.decoder_blocks = nn.ModuleList([
                Block(self.hidden_size, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
                for i in range(decoder_depth)])

            self.decoder_norm = nn.LayerNorm(self.hidden_size)
            self.decoder_pred = nn.Linear(self.hidden_size, self.patch_size**2 * 3, bias=True) # decoder to patch

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size), cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            fwd_params = sum(p.numel() for p in self.decoder_blocks.parameters() if p.requires_grad)
            dis_print(f"Fwd Decoder Parameters: {fwd_params / 1000000:.2f}M")
        if self.progress_pred:
            self.progress_queries=nn.Embedding(1, self.hidden_size) # [PROG]
            # progress decoder
            self.proj_progress_output_embed = nn.Linear(self.hidden_size, self.progress_decoder_dim)
            progress_decoder_depth = 2
            self.decode_progress = nn.ModuleList([
                Block(self.progress_decoder_dim, 8, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
                for i in range(progress_decoder_depth)])
            self.progress_mask_token_embedding = nn.Embedding(1, self.progress_decoder_dim)
            self.pred_progress = nn.Linear(self.progress_decoder_dim, 1) # pred progress  
            self.sigmoid_progress=nn.Sigmoid()

    def encode_texts(self, texts):
        inputs = self.text_tokenizer(texts)
        device = next(self.text_encoder.parameters()).device
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder.encode_text(inputs.to(device))
        return encoder_hidden_states
        
    def forward(self, input_dict, is_training=True):
        goal_rgb = input_dict['goal_rgb']  # (b, c, h, w)
        rgb = input_dict['rgb']  # (b, l, c, h, w)
        hand_rgb = input_dict['hand_rgb']  # (b, l, c, h, w)
        attention_mask = input_dict['attention_mask']  # (b, l)
        text=input_dict["text"][0] 
        progress_targets=input_dict["progress"]  #(b,l,)

        obs_preds = None
        obs_hand_preds = None
        obs_target = None
        obs_hand_target = None
        arm_action_preds = None
        gripper_action_preds = None
        action_mu_preds = None
        action_logvar_preds = None
        progress_preds=None

        batch_size, seq_length, c, h, w = rgb.shape


        if self.use_state:
            arm_state = input_dict['arm_state']  # (b, l, state_dim - 1)
            gripper_state = input_dict['gripper_state']  # (b, l, 2)
            arm_state_embeddings = self.embed_arm_state(arm_state.view(batch_size, seq_length, self.state_dim-1))  # (b, l, h)
            gripper_state_embeddings = self.embed_gripper_state(gripper_state)  # (b, l, h)
            state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)  # (b, l, 2h)
            state_embeddings = self.embed_state(state_embeddings)  # (b, l, h)

        # goal rgb mae feature
        goal_obs_embeddings, goal_patch_embeddings = self.model_mae(goal_rgb)  # (b, img_feat_dim), (b, 196, patch_feat_dim)
        
        # rgb mae feature
        obs_embeddings, patch_embeddings = self.model_mae(
            rgb.view(batch_size*seq_length, c, h, w)) # (b * l, img_feat_dim), (b * l, 196, patch_feat_dim)
        obs_embeddings = obs_embeddings.view(batch_size, seq_length, -1) # (b, l, img_feat_dim)

        # hand rgb mae feature
        if self.use_hand_rgb:
            hand_obs_embeddings, hand_patch_embeddings = self.model_mae(
                hand_rgb.view(batch_size*seq_length, c, h, w)) # (b * l, img_feat_dim), (b * l, 196, patch_feat_dim)
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, seq_length, -1) # (b, l, img_feat_dim)
        
        # compute obs target
        if self.fwd_pred:
            p = self.patch_size
            h_p = h // p
            w_p = w // p
            rgb = rgb.reshape(shape=(batch_size, seq_length, 3, h_p, p, w_p, p)) # b,len,3,14,p,14,p
            obs_target = rgb.permute(0, 1, 3, 5, 4, 6, 2) # b,len,14,14,p,p,3
            obs_target = obs_target.reshape(shape=(batch_size, seq_length, h_p * w_p, (p**2) * 3)) # b,len,14x14,p*p*3
            if not self.without_norm_pix_loss:
                # norm the target 
                obs_target = (obs_target - obs_target.mean(dim=-1, keepdim=True)
                    ) / (obs_target.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
            
            if self.fwd_pred_hand:
                hand_rgb = hand_rgb.reshape(shape=(batch_size, seq_length, 3, h_p, p, w_p, p)) # b,len,3,14,p,14,p
                obs_hand_target = hand_rgb.permute(0, 1, 3, 5, 4, 6, 2) # b,len,14,14,p,p,3
                obs_hand_target = obs_hand_target.reshape(shape=(batch_size, seq_length, h_p * w_p, (p**2)*3))
                if not self.without_norm_pix_loss:
                    # norm the target 
                    obs_hand_target = (obs_hand_target - obs_hand_target.mean(dim=-1, keepdim=True)
                        ) / (obs_hand_target.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)            

        if self.use_resampler:
            goal_patch_embeddings = goal_patch_embeddings.unsqueeze(1)  # (b, 1, 196, patch_feat_dim)
            goal_patch_embeddings = self.perceiver_resampler(goal_patch_embeddings)  # (b, 1, 9, patch_feat_dim)
            goal_patch_embeddings = goal_patch_embeddings.squeeze(1)  # (b, 9, patch_feat_dim)

            patch_embeddings = patch_embeddings.unsqueeze(1)  # (b * l, 1, 196, patch_feat_dim)
            patch_embeddings = self.perceiver_resampler(patch_embeddings)  # (b * l, 1, 9, patch_feat_dim)
            patch_embeddings = patch_embeddings.squeeze(1)  # (b * l, 9, patch_feat_dim)
            patch_embeddings = patch_embeddings.view(batch_size, seq_length, self.n_patch_latents, self.patch_feat_dim)  # (b, l, 9, patch_feat_dim)

            if self.use_hand_rgb:
                hand_patch_embeddings = hand_patch_embeddings.unsqueeze(1)  # (b * l, 1, 196, patch_feat_dim)
                hand_patch_embeddings = self.perceiver_resampler(hand_patch_embeddings)  # (b * l, 1, 9, patch_feat_dim)
                hand_patch_embeddings = hand_patch_embeddings.squeeze(1)  # (b * l, 9, patch_feat_dim)
                hand_patch_embeddings = hand_patch_embeddings.view(batch_size, seq_length, self.n_patch_latents, self.patch_feat_dim)  # (b, l, 9, patch_feat_dim)
        else:
            raise NotImplementedError
        

        # Embed language
        lang_embeddings = self.encode_texts(text)
        lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6) # normalization 
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)

        # embed images and patches
        goal_obs_embeddings = self.embed_goal_image(goal_obs_embeddings)  # (b, h)
        goal_patch_embeddings = self.embed_goal_patch(goal_patch_embeddings)  # (b, 9, h)
        obs_embeddings = self.embed_img(obs_embeddings)  # (b, l, h)
        patch_embeddings = self.embed_patch(patch_embeddings)  # (b, l, 9, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = self.embed_hand_img(hand_obs_embeddings)  # (b, l, h)
            hand_patch_embeddings = self.embed_hand_patch(hand_patch_embeddings)  # (b, l, 9, h)
        
        # add timestep embeddings
        time_embeddings = self.embed_timestep.weight # (l, h)

        lang_embeddings = lang_embeddings.view(batch_size, 1, -1).repeat(1,seq_length,1) + time_embeddings# 注意debug
        patch_embeddings = patch_embeddings + time_embeddings.view(seq_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings + time_embeddings

        
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings + time_embeddings
            hand_patch_embeddings = hand_patch_embeddings + time_embeddings.view(seq_length, 1, self.hidden_size)
        if self.use_state:
            state_embeddings = state_embeddings + time_embeddings

        # Format sequence: lang, state, patch, obs, hand_patch, hand_obs, [ACT], [OBS], [OBS_HAND],[PROG]
        lang_embeddings = lang_embeddings.view(batch_size, seq_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings.view(batch_size, seq_length, 1, self.hidden_size)
        if self.use_state:
            state_embeddings = state_embeddings.view(batch_size, seq_length, 1, self.hidden_size)
            stacked_inputs = torch.cat((lang_embeddings,state_embeddings, patch_embeddings, obs_embeddings), dim=2)  # (b, l, n_tokens, h)
        else:
            stacked_inputs = torch.cat((lang_embeddings,patch_embeddings, obs_embeddings), dim=2)  # (b, l, n_tokens, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, seq_length, 1, self.hidden_size)
            stacked_inputs = torch.cat((stacked_inputs, hand_patch_embeddings, hand_obs_embeddings), dim=2)  # (b, l, n_tokens, h)
        if self.act_pred:
            action_queries = self.action_queries.weight  # (1, h)
            action_queries = action_queries.view(1, 1, 1, self.hidden_size).repeat(batch_size, seq_length, 1, 1)  # (b, l, 1, h)
            stacked_inputs = torch.cat((stacked_inputs, action_queries), dim=2)  # (b, l, n_tokens, h)
        if self.fwd_pred:
            obs_queries = self.obs_queries.weight  # (10, h)
            obs_queries = obs_queries.view(1, 1, self.n_patch_latents + 1, self.hidden_size).repeat(batch_size, seq_length, 1, 1)
            stacked_inputs = torch.cat((stacked_inputs, obs_queries), dim=2)
            if self.fwd_pred_hand:
                obs_hand_queries = self.obs_hand_queries.weight  # (10, h)
                obs_hand_queries = obs_hand_queries.view(1, 1, self.n_patch_latents + 1, self.hidden_size).repeat(batch_size, seq_length, 1, 1)
                stacked_inputs = torch.cat((stacked_inputs, obs_hand_queries), dim=2)  # (b, l, n_tokens, h)
        if self.progress_pred:
            progress_queries = self.progress_queries.weight  # (1, h)
            progress_queries = progress_queries.view(1, 1, 1, self.hidden_size).repeat(batch_size, seq_length, 1, 1)  # (b, l, 1, h)
            stacked_inputs = torch.cat((stacked_inputs, progress_queries), dim=2)  # (b, l, n_tokens, h)
        # number of tokens for different modalities
        n_lang_tokens = 1
        n_state_tokens = 1
        n_patch_tokens = self.n_patch_latents
        n_obs_tokens = 1
        n_hand_patch_tokens = self.n_patch_latents
        n_hand_obs_tokens = 1
        n_act_pred_tokens = 1
        n_fwd_pred_tokens = n_patch_tokens + n_obs_tokens
        n_fwd_pred_hand_tokens = n_patch_tokens + n_obs_tokens
        n_progress_pred_tokens=1
        # compute number of tokens (does not include the conditioned goal image tokens)
        n_tokens = n_lang_tokens  
        if self.use_state:
            n_tokens += n_state_tokens
        n_tokens += n_patch_tokens 
        n_tokens += n_obs_tokens
        if self.use_hand_rgb:
            n_tokens += n_hand_obs_tokens
            n_tokens += n_hand_patch_tokens
        if self.act_pred:
            act_pred_token_i = n_tokens
            n_tokens += n_act_pred_tokens
        if self.fwd_pred:
            obs_pred_token_i = n_tokens
            n_tokens += n_fwd_pred_tokens
            if self.fwd_pred_hand:
                obs_pred_hand_token_i = n_tokens
                n_tokens += n_fwd_pred_hand_tokens
        if self.progress_pred:
            progress_pred_token_i = n_tokens
            n_tokens += n_progress_pred_tokens
        # number of condtioned tokens (goal image)
        n_condtioned_tokens = 1 + self.n_patch_latents
        
        # add goal image conditions at the front
        stacked_inputs = stacked_inputs.reshape(batch_size, n_tokens * seq_length, self.hidden_size)
        goal_obs_embeddings = goal_obs_embeddings.view(batch_size, 1, self.hidden_size)
        stacked_inputs = torch.cat((goal_patch_embeddings, goal_obs_embeddings, stacked_inputs), dim=1)  # (b, l * n_tokens + n_patch_latents + 1, h)
        assert stacked_inputs.shape == (batch_size, seq_length * n_tokens + n_condtioned_tokens, self.hidden_size)

        # layer norm
        stacked_inputs = self.embed_ln(stacked_inputs)

        # generate attention mask
        attn_mask = attention_mask.view(batch_size, 1, seq_length)
        lang_attn_mask=attn_mask.repeat(1, n_lang_tokens, 1)
        state_attn_mask = attn_mask.repeat(1, n_state_tokens, 1)
        patch_attn_mask = attn_mask.repeat(1, n_patch_tokens, 1)
        obs_attn_mask   = attn_mask.repeat(1, n_obs_tokens, 1)
        hand_patch_attn_mask = attn_mask.repeat(1, n_hand_patch_tokens, 1)
        hand_obs_attn_mask = attn_mask.repeat(1, n_hand_obs_tokens, 1)

        if self.use_state:
            stacked_attn_mask = torch.cat((lang_attn_mask,state_attn_mask, patch_attn_mask, obs_attn_mask), dim=1)
        else:
            stacked_attn_mask = torch.cat((lang_attn_mask,patch_attn_mask, obs_attn_mask), dim=1)
        if self.use_hand_rgb:
            stacked_attn_mask = torch.cat((stacked_attn_mask, hand_patch_attn_mask, hand_obs_attn_mask), dim=1)
        if self.act_pred:
            act_pred_attn_mask = torch.zeros((batch_size, n_act_pred_tokens, seq_length), dtype=torch.long).cuda()
            stacked_attn_mask = torch.cat((stacked_attn_mask, act_pred_attn_mask), dim=1)
        if self.fwd_pred:
            fwd_pred_attn_mask = torch.zeros((batch_size, n_fwd_pred_tokens, seq_length), dtype=torch.long).cuda()
            stacked_attn_mask = torch.cat((stacked_attn_mask, fwd_pred_attn_mask), dim=1)
            if self.fwd_pred_hand:
                fwd_pred_hand_attn_mask = torch.zeros((batch_size, n_fwd_pred_hand_tokens, seq_length), dtype=torch.long).cuda()
                stacked_attn_mask = torch.cat((stacked_attn_mask, fwd_pred_hand_attn_mask), dim=1)
        if self.progress_pred:
            progress_pred_attn_mask = torch.zeros((batch_size, n_progress_pred_tokens, seq_length), dtype=torch.long).cuda()
            stacked_attn_mask = torch.cat((stacked_attn_mask, progress_pred_attn_mask), dim=1)

        stacked_attn_mask = stacked_attn_mask.permute(0, 2, 1)  # (b, l, n_tokens)
        stacked_attn_mask = stacked_attn_mask.reshape(batch_size, n_tokens * seq_length)  # (b, l * n_tokens)
        goal_obs_attn_mask = torch.ones((batch_size, 1), dtype=torch.long).cuda()
        goal_patch_attn_mask = torch.ones((batch_size, self.n_patch_latents), dtype=torch.long).cuda()
        stacked_attn_mask = torch.cat((goal_patch_attn_mask, goal_obs_attn_mask, stacked_attn_mask), dim=1)  # (b, l * n_tokens + n_patch_latens + 1)
        assert stacked_attn_mask.shape == (batch_size, seq_length * n_tokens + n_condtioned_tokens)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attn_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x[:, n_condtioned_tokens:]
        x = x.reshape(batch_size, seq_length, n_tokens, self.hidden_size)  # (b, l, n_tokens, h)

        # action prediction: predict next action given obs
        # format sequence
        if self.act_pred:
            action_output_embedding = x[:, :, act_pred_token_i]  # (b, l, h)

            # encode action
            arm_action = input_dict['arm_action'] # b,len,act_len,act_dim-1
            gripper_action = input_dict['gripper_action'].long() # b,len,act_len
            arm_action_embeddings = self.embed_arm_action(arm_action) # b,len,act_len,act_encoder_dim
            gripper_action_embeddings = self.embed_gripper_action(gripper_action) # b,len,act_len,act_encoder_dim
            action_embeddings = torch.cat((arm_action_embeddings, gripper_action_embeddings), dim=-1) # b,len,act_len,2*act_encoder_dim
            action_embeddings = self.embed_action(action_embeddings) # b,len,act_len,act_encoder_dim
            cls_token_embeddings = self.action_encoder_cls_token.weight # 1,act_encoder_dim
            cls_token_embeddings = cls_token_embeddings.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_length, 1, 1) # b,len,1,act_encoder_dim
            z = torch.cat((cls_token_embeddings, action_embeddings), dim=2) # b,len,1+act_len,act_encoder_dim
            action_encoder_positional_embeddings = self.action_encoder_positional_embeddings.weight # 1+act_len,act_encoder_dim
            z = z + action_encoder_positional_embeddings # b,len,1+act_len,act_encoder_dim
            z = z.reshape(batch_size * seq_length, self.act_len + 1, self.act_encoder_dim) # b*len,1+1+act_len,act_encoder_dim
            for blk in self.encode_action:
                z = blk(z)
            action_latent_embedding = z[:, 0] # b*len,act_encoder_dim
            action_latent_embedding = action_latent_embedding.reshape(batch_size, seq_length, self.act_encoder_dim) # b,len,act_encoder_dim
            action_latent_preds = self.pred_style_vector(action_latent_embedding) # b,len,2*act_latent_dim
            action_mu_preds = action_latent_preds[:, :, :self.act_latent_dim] # b,len,act_latent_dim
            action_logvar_preds = action_latent_preds[:, :, self.act_latent_dim:] # b,len,act_latent_dim
            # sample style vector
            action_mu_preds = action_mu_preds.view(-1, self.act_latent_dim)
            action_logvar_preds = action_logvar_preds.view(-1, self.act_latent_dim)
            action_style_vector = reparameterize(action_mu_preds, action_logvar_preds) # b*len,act_latent_dim
            action_style_vector = action_style_vector.view(batch_size, seq_length, self.act_latent_dim) # b,len,act_latent_dim
            if not is_training: # we set the mean=0 and var=1 during inference
                action_style_vector = torch.zeros([batch_size, seq_length, self.act_latent_dim], dtype=torch.float32).to(rgb.device)
                action_style_vector = action_style_vector.type_as(arm_action)
            action_style_embeddings = self.embed_style_vector(action_style_vector) # b,len,act_decoder_dimv   


            action_output_embedding = self.proj_action_output_embed(action_output_embedding)  # (b, l, act_decoder_dim)
            action_mask_token = self.action_mask_token_embedding.weight  # (1, act_decoder_dim)
            action_mask_token = action_mask_token.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_length, self.act_len, 1)  # (b, l, act_len, act_decoder_dim)
            action_output_embedding = action_output_embedding.view(batch_size, seq_length, 1, self.act_decoder_dim)     
            action_decoder_positional_embeddings = self.action_decoder_positional_embeddings.weight

            action_style_embeddings = action_style_embeddings.view(batch_size, seq_length, 1, self.act_decoder_dim)
            y = torch.cat((action_style_embeddings, action_output_embedding, action_mask_token), dim=2)  # (b, l, 2 + act_len, act_decoder_dim)
            y[:, :, 2:] = y[:, :, 2:] + action_decoder_positional_embeddings
            y = y.reshape(batch_size * seq_length, 2 + self.act_len, self.act_decoder_dim)

            
            # forward transformer
            for blk in self.decode_action:
                y = blk(y)
            
     
            action_decoder_output_embeddings = y[:, 2:]  # (b * l, act_len, act_decoder_dim)



            action_decoder_output_embeddings = action_decoder_output_embeddings.reshape(batch_size, seq_length, self.act_len, self.act_decoder_dim)
            arm_action_preds = self.pred_arm_act(action_decoder_output_embeddings)  # (b, l, act_len, act_dim - 1)
            gripper_action_preds = self.pred_gripper_act(action_decoder_output_embeddings)  # (b, l, act_len, 1)
                
        # forward prediction: predict next obs
        if self.fwd_pred:
            mask_token = self.mask_token  # (1, 1, 1, h)
            mask_tokens = mask_token.repeat(batch_size, seq_length, (self.image_size//self.patch_size)**2, 1)  # (b, l, n_patches, h)
            mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(0).repeat(batch_size, seq_length, 1, 1)  # (b, l, n_patch, h)

            obs_pred = self.decoder_embed(x[:, :, obs_pred_token_i:(obs_pred_token_i + n_fwd_pred_tokens)])  # (b, l, n_patch_latents + 1, h)
            obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2)  # (b, l, n_patch + n_patch_latens + 1, h)
            obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])  # (b * l, n_patch + n_patch_latens + 1, h)
            for blk in self.decoder_blocks:
                obs_pred_ = blk(obs_pred_)
            obs_pred_ = self.decoder_norm(obs_pred_)
            obs_preds = self.decoder_pred(obs_pred_)  # (b * l, n_patch + n_patch_latens + 1, h)
            obs_preds = obs_preds.reshape(batch_size, seq_length, -1, obs_preds.shape[-1])  # (b, l, n_patch + n_patch_latens + 1, h)
            obs_preds = obs_preds[:, :, n_fwd_pred_tokens:]  # (b, len, n_patch, h)

            if self.fwd_pred_hand:
                obs_pred_hand = self.decoder_embed(x[:, :, obs_pred_hand_token_i:(obs_pred_hand_token_i + n_fwd_pred_hand_tokens)])  # (b, l, n_patch_latents + 1, h)
                obs_pred_hand_ = torch.cat([obs_pred_hand, mask_tokens], dim=2)  # (b, l, n_patch + n_patch_latens + 1, h)
                obs_pred_hand_ = obs_pred_hand_.reshape(-1, obs_pred_hand_.shape[-2], obs_pred_hand_.shape[-1])  # (b * l, n_patch + n_patch_latens + 1, h)
                for blk in self.decoder_blocks:
                    obs_pred_hand_ = blk(obs_pred_hand_)
                obs_pred_hand_ = self.decoder_norm(obs_pred_hand_)
                obs_hand_preds = self.decoder_pred(obs_pred_hand_)  # (b * l, n_patch + n_patch_latens + 1, h)
                obs_hand_preds = obs_hand_preds.reshape(batch_size, seq_length, -1, obs_hand_preds.shape[-1]) # (b, l, n_patch + n_patch_latens + 1, h)
                obs_hand_preds = obs_hand_preds[:, :, n_fwd_pred_hand_tokens:] # (b, l, n_patch, h)
        
        # progress prediction
        # format sequence
        if self.progress_pred:
            progress_output_embedding = x[:, :, progress_pred_token_i]  # (b, l, h)
            progress_output_embedding = self.proj_progress_output_embed(progress_output_embedding)  # (b, l, progress_decoder_dim)
            progress_mask_token = self.progress_mask_token_embedding.weight  # (1, progress_decoder_dim)
            progress_mask_token = progress_mask_token.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_length, 1, 1)  # (b, l, 1, progress_decoder_dim)
            progress_output_embedding = progress_output_embedding.view(batch_size, seq_length, 1, self.progress_decoder_dim)     
            y = torch.cat((progress_output_embedding, progress_mask_token), dim=2)  # (b, l, 1 + 1, progress_decoder_dim)
            y = y.reshape(batch_size * seq_length, 1 + 1, self.act_decoder_dim)
            
            # forward transformer
            for blk in self.decode_progress:
                y = blk(y)
            
            # get output
            progress_decoder_output_embeddings = y[:, 1:]  # (b * l, 1, progress_decoder_dim)
            progress_decoder_output_embeddings = progress_decoder_output_embeddings.reshape(batch_size, seq_length, 1, self.progress_decoder_dim)
            progress_preds = self.pred_progress(progress_decoder_output_embeddings).squeeze()  # (b, l)
            progress_preds=self.sigmoid_progress(progress_preds)
            



        prediction = {
            'obs_preds': obs_preds,
            'obs_target': obs_target,
            'obs_hand_preds': obs_hand_preds,
            'obs_hand_target': obs_hand_target,
            'arm_action_preds': arm_action_preds,
            'gripper_action_preds': gripper_action_preds,
            'action_mu_preds': action_mu_preds,
            'action_logvar_preds': action_logvar_preds,
            'progress_preds':progress_preds,
            'progress_targets':progress_targets,
        }
        return prediction
    

    def evaluate(self, input_dict,original_gripper=False,return_progress=False):

        attention_mask = input_dict['attention_mask']
        prediction = self.forward(input_dict, is_training=False)

        arm_action_preds = prediction['arm_action_preds'] # (1, len, act_len, act_dim-1)
        gripper_action_preds = prediction['gripper_action_preds'] # (1, len, act_len, 1)
        


        arm_action_preds = arm_action_preds.squeeze(0) # (len, act_len, act_dim-1)
        gripper_action_preds = gripper_action_preds.squeeze() # (len, act_len)
        arm_action_preds = arm_action_preds[attention_mask.flatten() > 0]
        gripper_action_preds = gripper_action_preds[attention_mask.flatten() > 0]


        # Take the last action
        arm_action_pred = arm_action_preds[-1].cpu() # (act_len, act_dim-1)
        arm_action_pred = arm_action_pred[0] # (act_dim-1, )
        gripper_action_pred = gripper_action_preds[-1:].cpu() # (1, act_len)
        gripper_action_pred = gripper_action_pred[:, 0] # (1, 1)

        if original_gripper:
            gripper_action_pred = torch.nn.Sigmoid()(gripper_action_pred)
        else:
            gripper_action_pred = torch.nn.Sigmoid()(gripper_action_pred)
            gripper_action_pred = gripper_action_pred > 0.5
            gripper_action_pred = gripper_action_pred.int().float()
            gripper_action_pred = gripper_action_pred * 2.0 - 1.0
        action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=0) # (act_dim,)
        if return_progress:
            progress=prediction["progress_preds"]
            progress=progress[-1]
            return action_pred,progress
        else:
            return action_pred