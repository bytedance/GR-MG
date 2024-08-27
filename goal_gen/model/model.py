# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from transformers import T5Tokenizer, T5EncoderModel
import os
#  modified from https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py
class IP2P(nn.Module):
    """InstructPix2Pix model."""
    def __init__(self, 
                 pretrained_model_dir,
                 device,
                 seed=123,
                 conditioning_dropout_prob=None,
                 gradient_checkpointing=False):
        super().__init__()
        self.device = device
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_dir, subfolder="scheduler")
        text_encoder_name = "t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name) 
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_dir, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_dir, subfolder="unet")
        
        # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
        # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
        # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
        # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
        # initialized to zero.
        self.in_channels = 8
        self.unet.register_to_config(in_channels=self.in_channels)
        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        

        # Conditioning dropout probability  used for classifier free guidance
        self.conditioning_dropout_prob = conditioning_dropout_prob

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing() # it will reduce GPU memory but add computing burden

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def tokenize_texts(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=77)
        return inputs

    def forward(self, input_dict):
        original_pixel_values = input_dict['original_pixel_values']
        edited_pixel_values = input_dict['edited_pixel_values']
        input_text = input_dict['input_text'][0]
        progress=input_dict["progress"]*10#(b,)
        input_text=[  text+f".And {curr_progress}% of the instruction has been finished." for (text,curr_progress) in zip(input_text,progress)]
        input_ids=self.tokenize_texts(input_text)

        # We want to learn the denoising process w.r.t the edited images which
        # are conditioned on the original image (which was edited) and the edit instruction.
        # So, first, convert images to latent space.
        latents = self.vae.encode(edited_pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,)).to(latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(**(input_ids.to(self.device))).last_hidden_state

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        original_image_embeds = self.vae.encode(original_pixel_values).latent_dist.mode()

        # Conditioning dropout to support classifier-free guidance during inference.
        if self.conditioning_dropout_prob is not None:
            random_p = torch.rand(bsz, device=latents.device, generator=self.generator)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * self.conditioning_dropout_prob
            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            # Final text conditioning.
            null_conditioning = self.text_encoder(**(self.tokenize_texts([""]).to(self.device))).last_hidden_state
            encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

            # Sample masks for the original images.
            image_mask_dtype = original_image_embeds.dtype
            image_mask = 1 - (
                (random_p >= self.conditioning_dropout_prob).to(image_mask_dtype)
                * (random_p < 3 * self.conditioning_dropout_prob).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            # Final image conditioning.
            original_image_embeds = image_mask * original_image_embeds

            # Concatenate the `original_image_embeds` with the `noisy_latents`.
            concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

            # Get the target for loss depending on the prediction type
            target = noise

            # Predict the noise residual and compute loss
            prediction = self.unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            return prediction, target
