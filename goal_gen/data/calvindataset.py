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
import json
import random
from tracemalloc import is_tracing

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

class CalvinDataset_Goalgen(Dataset):
    def __init__(self,
                 data_dir,
                 resolution=256,
                 resolution_before_crop=288,
                 center_crop=False,
                 forward_n_min_max=[20, 22],
                 use_full=True,
                 is_training=True,
                 color_aug=False,
    ):
        super().__init__()
        self.is_training = is_training
        self.color_aug=color_aug # whether to use ColorJitter
        self.center_crop=center_crop # whether to use CenterCrop
        if is_training:
            self.data_dir = os.path.join(data_dir, "training") 
        else:
            self.data_dir = os.path.join(data_dir, "validation")

        self.forward_n_min, self.forward_n_max = forward_n_min_max
        self.use_full = use_full #whether to use every frame in a trajectory
        self.resolution = resolution
        self.resolution_before_crop = resolution_before_crop

        # image preprocessing
        if self.is_training:
            self.transform = transforms.Compose(
                    [
                        transforms.Resize((self.resolution_before_crop, self.resolution_before_crop)),
                        transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
                    ]
                )                
        else:
            self.transform = transforms.Resize((self.resolution, self.resolution))

        # to improve the speed of loading data , we first preprocess the calvin dataset
        # you should first run utils/format_calvin_data.py to get meta.json
        meta_json = os.path.join(self.data_dir, "meta.json")

        with open(meta_json, "r") as f:
            self.meta = json.load(f)

        n_trajs = len(self.meta.keys())
        self.sample_tuples = []
        for traj_id in self.meta.keys():
            n_frames = self.meta[traj_id]['num_frames']
            for frame_id in range(n_frames):
                if self.use_full:
                    temp_max = 1
                else:
                    temp_max = self.forward_n_max
                if (frame_id + temp_max) < n_frames:
                    sample_tuple = (traj_id, frame_id, n_frames)
                    self.sample_tuples.append(sample_tuple)
        
        print(f"=" * 20)
        print(f'{len(self)} samples in total...')
        print(f'{n_trajs} trajectories in total...')
    
    def __len__(self):
        return len(self.sample_tuples)

    def __getitem__(self, index):
        if not self.is_training:
            np.random.seed(index)
            random.seed(index)
        
        sample_tuple = self.sample_tuples[index]
        traj_id, frame_id, n_frames = sample_tuple

        # text
        edit_prompt= self.meta[traj_id]['text']

        # input image
        input_image_path = os.path.join(self.data_dir, f"{traj_id}", f"{frame_id}_static.png")
        input_image = Image.open(input_image_path).convert("RGB")

        # goal image
        forward_n = random.choice(range(self.forward_n_min, self.forward_n_max + 1))
        edited_frame_id = min(frame_id + forward_n, n_frames - 1)
   

        assert edited_frame_id < n_frames
        edited_image_path = os.path.join(self.data_dir, f"{traj_id}", f"{edited_frame_id}_static.png")
        edited_image = Image.open(edited_image_path).convert("RGB")
        
        bright_range=random.uniform(0.8,1.2)
        contrast_range=random.uniform(0.8,1.2)
        saturation_range=random.uniform(0.8,1.2)
        hue_range=random.uniform(-0.04,0.04)
        if self.is_training and self.color_aug and random.random()>0.4:  # apply color jitter with probability of 0.6
            self.color_trans = transforms.Compose(
                    [
                        transforms.ColorJitter(brightness=(bright_range,bright_range), contrast=(contrast_range,contrast_range), saturation=(saturation_range,saturation_range), hue=(hue_range,hue_range)), 
                    ]
                )    
            input_image=self.color_trans(input_image)
            edited_image=self.color_trans(edited_image) # apply the same transformation to input image and goal image
        # preprocess_images
        concat_images = np.concatenate([np.array(input_image), np.array(edited_image)], axis=2)
        concat_images = torch.tensor(concat_images)
        concat_images = concat_images.permute(2, 0, 1)
        concat_images = 2 * (concat_images / 255) - 1


        concat_images = self.transform(concat_images)

        input_image, edited_image = concat_images.chunk(2)

        input_image = input_image.reshape(3, self.resolution, self.resolution)
        edited_image = edited_image.reshape(3, self.resolution, self.resolution)

        example = dict()
        example['input_text'] = [edit_prompt]
        example['original_pixel_values'] = input_image
        example['edited_pixel_values'] = edited_image
        index=int((frame_id+1)/n_frames*10)
        if index==10:
            index=9
        example["progress"]= index
        return example


if __name__ == "__main__":
    dataset = CalvinDataset_Goalgen(
        data_dir="PATH_TO_CALVIN/calvin/task_ABC_D", 
        resolution=256,
        resolution_before_crop=288,
        center_crop=True,
        forward_n_min_max=(20, 22),  # FIXME: hardcode
        is_training=True,
        use_full=True,
        color_aug=True
    )

    for i in tqdm(range(0, len(dataset),30)):
        example = dataset[i]
        original_pixel_values = example['original_pixel_values']
        edited_pixel_values = example['edited_pixel_values']
        progress=example["progress"]
        text=example['input_text']
        print(text)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        original_image = original_pixel_values.permute(1, 2, 0).numpy()
        original_image = (original_image + 1) / 2 * 255
        original_image = np.clip(original_image, 0, 255)
        original_image = original_image.astype(np.uint8)
        ax[0].imshow(original_image)

        edited_image = edited_pixel_values.permute(1, 2, 0).numpy()
        edited_image = (edited_image + 1) / 2 * 255
        edited_image = np.clip(edited_image, 0, 255)
        edited_image = edited_image.astype(np.uint8)
        ax[1].imshow(edited_image)
        plt.savefig("debug.png", dpi=300)