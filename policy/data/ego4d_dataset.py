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
"""
Code for loading ego4d video clip data for pretraining.
This dataset contains language + video.
Return: text, image sequence, attention_mask
"""

from __future__ import annotations
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TVF
from decord import VideoReader, cpu, gpu
import matplotlib.pyplot as plt

# source: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

class RandomShiftsSingleAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Ego4DDataset_Policy(Dataset):
    def __init__(self,
                 data_dir,
                 preprocess=None,
                 video_sample_rate=2,
                 seq_len=10,
                 mode='train',
                 annotation_file='train800k.json',
                 use_data_augmentation=False,
                 goal_interval=7):

        """Constructor.
        Args:
            data_dir: root directory of the data
            preprocess: image preprocess function
            seq_len: sequence length
            mode: 'train' ,'val'
            use_data_augmentation: whether to use data augmentation
        """
        super().__init__()
        self.dataset_dir = data_dir
        self.preprocess = preprocess
        self.video_sample_rate = video_sample_rate
        self.seq_len = seq_len
        self.mode = mode
        self.annotation_file = annotation_file
        self.use_data_augmentation = use_data_augmentation
        self.video_names, self.video_texts = self.get_annotations()
        self.goal_interval=goal_interval
        self.state_dim = 7
        self.action_dim = 7

        self.input_size = (224, 224)
        self.clip_mean = (0.485, 0.456, 0.406)
        self.clip_std = (0.229, 0.224, 0.225)


        self.preprocess_goal=T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        
        if self.use_data_augmentation:
            self.preprocess_obs = T.Compose([
                RandomShiftsSingleAug(pad=10), 
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        else:
            self.preprocess_obs = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        self.act_len=10 # Ego4D does not include actions, we keep it to align with calvin_dataset file
        
    def get_annotations(self):
        json_path = os.path.join(self.annotation_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        video_names = data['video_name']
        video_texts = data['text']
        assert len(video_names) == len(video_texts)
        return video_names, video_texts 

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        # text
        text = self.video_texts[index]

        # video
        video_name = self.video_names[index]
        video_path = os.path.join(self.dataset_dir, video_name)
        # the resolution of original Ego4D is 1080*1440. we resize it to 224*224 in advance to improve the loading speed
        video_full = VideoReader(video_path, num_threads=16, ctx=cpu(0)) 

        length=len(video_full)
        if length > self.seq_len*self.video_sample_rate:
            start = np.random.choice(range(0, length-self.seq_len*self.video_sample_rate))
        else:
            start=0
        all_index = [i * (self.video_sample_rate)+start for i in range(self.seq_len) 
            if (i * self.video_sample_rate+start) < length]
        goal_frame_index=min(length-1,all_index[-1]+self.goal_interval)
        all_index.append(goal_frame_index)
        frames = video_full.get_batch(all_index).asnumpy()        
        frames = [TVF.to_tensor(frame) for frame in frames]
        goal_frame=frames.pop()
        assert len(frames)==(len(all_index)-1)
        end = start + (len(frames)-1)*self.video_sample_rate
        frames_tensor = torch.stack(frames, 0)
        # pre-process  each frame
        static_rgbs = self.preprocess_obs(frames_tensor)
        goal_rgb=self.preprocess_goal(goal_frame)

        # RGB
        tlen, C, H, W = static_rgbs.shape
        rgb_data = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        rgb_data[:tlen] = static_rgbs
        goal_rgb_data=goal_rgb.to(rgb_data.dtype)

        # Attention mask (should be all 1 for full dataset)
        attention_mask = np.ones(self.seq_len, dtype=np.int32) # (len)
        attention_mask[tlen:] = 0.0
        attention_mask_data = torch.from_numpy(attention_mask).long()

        # To keep align with calvin_dataset file
        padded_states = torch.zeros(self.seq_len, self.state_dim).float() # (len, state_dim)
        padded_actions = torch.zeros(self.seq_len, self.act_len, self.action_dim).float() # (len, act_len, action_dim)
        padded_actions_mask = torch.zeros(self.seq_len, self.act_len, self.action_dim).float() # (len, act_len, action_dim)
        padded_hand_rgb= torch.zeros_like(rgb_data)
        padded_rel_states = torch.zeros(self.seq_len, self.state_dim).float()
        padded_progress_data= torch.zeros(self.seq_len).float()


        data = dict()
        data['goal_rgb'] = goal_rgb_data
        data['rgb'] = rgb_data # (len, C, H, W)
        data["hand_rgb"]=padded_hand_rgb
        data['state'] = padded_states # (len, state_dim)
        data['action'] = padded_actions # (len, action_dim)
        data['attention_mask'] = attention_mask_data # (len,)
        data['rel_state']= padded_rel_states
        data['action_mask']=padded_actions_mask
        data["text"]=[text]
        data["progress"]=padded_progress_data
        return data
  
if __name__ == "__main__":
    data_dir = '/PATH_TO_PRETRAIN_DATA/videos_rescale'
    DS0 = Ego4DDataset_Policy(
        data_dir=data_dir,
        preprocess=None, 
        video_sample_rate=2, 
        seq_len=10, 
        mode='train', 
        annotation_file="/PATH_TO_PRETRAIN_DATA/train800k.json",
        use_data_augmentation=False,
        goal_interval=7)

    length=len(DS0)
    for i in tqdm(range(length)):
        data=DS0[i]
        # continue
        rgb = data['rgb']
        goal_rgb=data["goal_rgb"]

        b = rgb.shape[0]
        # 设置画布大小
        fig, ax = plt.subplots(1, b+1, figsize=(b, 20), 
                                subplot_kw={'xticks': [], 'yticks': []},
                                gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
        for i in range(b):
            action = data['action'][i]
            text = data['text']
            print("text:",text)
            attention_mask = data['attention_mask'][i]
            temp_rgb = rgb[i].permute(1, 2, 0).numpy()
            temp_rgb = (temp_rgb + 1.0) * 127.5
            temp_rgb = np.clip(temp_rgb, 0.0, 255.0)
            temp_rgb = temp_rgb.astype(np.uint8)
            ax[i].imshow(temp_rgb)
        temp_rgb2 = goal_rgb.permute(1, 2, 0).numpy()
        temp_rgb2 = (temp_rgb2 + 1.0) * 127.5
        temp_rgb2 = np.clip(temp_rgb2, 0.0, 255.0)
        temp_rgb2 = temp_rgb2.astype(np.uint8)
        ax[b].imshow(temp_rgb2)
        plt.tight_layout(pad=0.0)
        plt.savefig(f"./debug.png")
        plt.close()