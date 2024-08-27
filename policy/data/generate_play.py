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
import numpy as np
from tqdm import tqdm
import json
def check_no_lan_frame(path):
    # return all the frames without language annotations
    anns = np.load(
    os.path.join(path, "lang_annotations", "auto_lang_ann.npy"), 
    allow_pickle=True).item()
    ann_number=set()
    n_trajs = len(anns['info']['indx'])
    for traj_idx in tqdm(range(n_trajs)):
        traj_st, traj_ed = anns['info']['indx'][traj_idx]
        for num in range(traj_st,traj_ed+1):
            ann_number.add(num)

    all_file=os.listdir(path)
    file_number=[]
    for file_name in all_file:
        if "episode" in file_name:
            if file_name[8:15]=="0000000":
                now_num=0
            else:
                now_num=int(file_name[8:15].lstrip('0'))
            if now_num not in ann_number:
                file_number.append(now_num)
    return file_number

def save_play_traj(path,file_number,save_name="play.json"):
    # return play trajectory 
    no_lan_traj=dict()
    no_lan_traj["st_ed_list"]=[]
    st=file_number[0]
    prev_num=file_number[0]
    count=0
    index=[]
    for num in tqdm(file_number[1:]):
        try:
            frame = np.load(os.path.join(path, f"episode_{num:07d}.npz"))
            action=frame['actions']
            rel_act=frame['rel_actions']
            robot_obs=frame['robot_obs']
            static_rgb = frame['rgb_static']
            hand_rgb = frame['rgb_gripper']
            if st==-1 and prev_num==-1:
                st=num
                prev_num=num
                continue
            if num-prev_num==1:
                prev_num=num 
            else:
                if st !=prev_num:
                    no_lan_traj["st_ed_list"].append((st,prev_num))
                st=num
                prev_num=num
        except:
            # we find the play data seems to have some bad files, so we add this protection mechanism
            count+=1
            print(f"bad case: {count}")
            index.append(num)
            print(f"bad index{num}")
            if st !=prev_num:
                    no_lan_traj["st_ed_list"].append((st,prev_num))
            st=-1
            prev_num=-1
    try:
        with open(os.path.join(path, save_name), "w") as f:
            json.dump(no_lan_traj, f)
    except:
        assert False


    print(f"bad case: {count}")
    print(f"bad index: ",index)  


if __name__ == "__main__":
    path=r"/PATH_TO_CALVIN/task_ABC_D/training"
    file_number=check_no_lan_frame(path)
    file_number.sort()
    save_play_traj(path,file_number,save_name="play.json")
