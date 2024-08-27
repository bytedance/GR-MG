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

# Format calvin data for training ip2p
import os
import argparse
import json
from tqdm import tqdm

import numpy as np
import cv2


SPLITS = ['training']
# SPLITS = ['validation']
def main(data_dir, target_dir, num_trajs_per_task=10000):
    for split in SPLITS:
        meta = dict()
        split_dir = os.path.join(target_dir, split)
        os.mkdir(split_dir)
        dataset_dir = os.path.join(data_dir, split)
        anns = np.load(
            os.path.join(dataset_dir, "lang_annotations", "auto_lang_ann.npy"), 
            allow_pickle=True).item()
        n_trajs = len(anns['info']['indx'])
        task_dict = {}
        for traj_idx in tqdm(range(n_trajs)):
            if split == 'training':
                # sample trajectories based on num_trajs_per_task
                traj_task = anns['language']['task'][traj_idx]
                if traj_task not in task_dict:
                    task_dict[traj_task] = 1
                else:
                    task_dict[traj_task] = task_dict[traj_task] + 1
                if task_dict[traj_task] > num_trajs_per_task:
                    continue

            traj_dir = os.path.join(split_dir, f"{traj_idx}")
            os.mkdir(traj_dir)
            traj_st, traj_ed = anns['info']['indx'][traj_idx]
            traj_text = anns['language']['ann'][traj_idx]
            for i in range(traj_st, traj_ed + 1):
                frame = np.load(os.path.join(dataset_dir, f"episode_{i:07d}.npz"))
                static_rgb = frame['rgb_static']
                hand_rgb = frame['rgb_gripper']
                cv2.imwrite(os.path.join(traj_dir, f"{i - traj_st}_static.png"), cv2.cvtColor(static_rgb, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(traj_dir, f"{i - traj_st}_hand.png"), cv2.cvtColor(hand_rgb, cv2.COLOR_BGR2RGB))
            meta[traj_idx] = {"text": traj_text, "num_frames": int(traj_ed - traj_st + 1)}
        with open(os.path.join(split_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", 
                        type=str,
                        default="",
                        help="data directory")
    parser.add_argument("--target_dir", 
                        type=str,
                        default="",
                        help="target data directory")
    parser.add_argument("--num_trajs_per_task",
                        type=int,
                        default=10000,  # when you want to do few-shot experiments, change this number
                        help="number of trajectories per task")
    args = parser.parse_args()

    main(args.data_dir, args.target_dir, args.num_trajs_per_task)