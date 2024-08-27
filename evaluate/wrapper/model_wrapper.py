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
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import policy.model.vision_transformer as vits
from utils.utils import euler2rotm, rotm2euler
from copy import deepcopy
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from policy.model.model import GR_MG
import time
import clip
GRIPPER_OPEN = 1
GRIPPER_CLOSE = 0
class CustomModel(CalvinBaseModel):
    def __init__(self,
                 ckpt_path,
                 configs,
                 device):
        self.device = device
        # model config
        self.configs = configs
        self.seq_len = configs["policy"]['seq_len']
        self.act_len = configs["policy"]["act_len"]
        self.device = device
        input_size = (224, 224)
        clip_mean = (0.485, 0.456, 0.406)
        clip_std = (0.229, 0.224, 0.225)
        self.preprocess = T.Compose([
            T.Resize(input_size, interpolation=Image.BICUBIC),
            T.Normalize(clip_mean, clip_std)])
        model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        model_mae.to(self.device)
        training_target = []
        if configs["trainer"]['act_pred']:
            training_target.append('act_pred')
        if configs["trainer"]['fwd_pred']:
            training_target.append('fwd_pred')
        if configs["trainer"]['fwd_pred_hand']:
            training_target.append('fwd_pred_hand')
        if configs["trainer"]["progress_pred"]:
            training_target.append('progress_pred')
        print(f"training target: {training_target}")

        #modify before release
        #clip model
        clip_name = "ViT-B/32"
        clip_model, clip_preprocess = clip.load(clip_name)
        # freeze clip
        for _, param in clip_model.named_parameters():
            param.requires_grad = False
        policy_config = self.configs['policy']
        input_config=self.configs["input"]
        trainer_config=self.configs["trainer"]
        self.policy = GR_MG(
                state_dim=input_config['state_dim'],
                act_dim=input_config['act_dim'],
                act_len=policy_config['act_len'],
                act_latent_dim=policy_config['act_latent_dim'],
                act_encoder_dim=policy_config['act_encoder_dim'],
                act_decoder_dim=policy_config['act_decoder_dim'],
                progress_decoder_dim=self.configs["policy"]["progress_decoder_dim"],
                hidden_size=policy_config['embed_dim'],
                model_mae=model_mae,
                clip_model=clip_model,
                img_feat_dim=policy_config["img_feat_dim"],
                lang_feat_dim = policy_config["lang_feat_dim"],
                patch_feat_dim=policy_config["patch_feat_dim"],
                resampler_params=policy_config['resampler_params'],
                max_length=policy_config['seq_len'],
                training_target=training_target,
                without_norm_pix_loss=trainer_config['without_norm_pix_loss'],
                use_hand_rgb=input_config['use_hand_rgb'],
                use_state=input_config['use_state'],
                use_resampler=policy_config['use_resampler'],
                n_layer=policy_config['n_layer'],
                n_head=policy_config['n_head'],
                n_inner=4*policy_config['embed_dim'],
                activation_function=policy_config['activation_function'],
                n_positions=1024,
                resid_pdrop=policy_config['dropout'],
                attn_pdrop=policy_config['dropout'],
                device=self.device)
        

        # Set up the model
        payload = torch.load(ckpt_path)
        epoch = payload['epoch']
        state_dict = payload['state_dict']
        print(f"loading state dict from epoch {epoch}...")

        del payload
        # Remove the prefix "model." from pl models
        pure_state_dict = dict()
        for k, v in state_dict.items():
            if "model." in k:
                new_k = k[6:]
                pure_state_dict[new_k] = v
        msg = self.policy.load_state_dict(pure_state_dict, strict=True)
        print(msg)
        self.policy.to(self.device)
        self.policy.eval()

    def reset(self):
        """Reset function."""
        self.rgb_list = []
        self.hand_rgb_list = []
        self.state_list = []
        self.rollout_step_counter = 0

    @staticmethod
    def compute_rel_state(states):
        first_xyz = states[0][0]
        first_rotm = states[0][1]
        first_gripper = states[0][2]
        seq_len = len(states)
        arm_states = np.zeros((seq_len, 6))
        gripper_states = np.zeros(seq_len)
        gripper_states[0] = first_gripper
        for i in range(1, seq_len):
            curr_xyz = states[i][0]
            curr_rotm = states[i][1]
            curr_gripper = states[i][2]
            rel_rotm = first_rotm.T @ curr_rotm
            rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
            arm_states[i, 0:3] = rel_xyz
            arm_states[i, 3:6] = rotm2euler(rel_rotm)
            gripper_states[i] = curr_gripper
        return arm_states, gripper_states

    def step(self, obs, goal, text):
        """Step function."""
        goal_rgb = goal[0]

        rgb = obs['rgb_obs']['rgb_static'] # (200, 200, 3)
        hand_rgb = obs['rgb_obs']['rgb_gripper']

        goal_rgb = Image.fromarray(goal_rgb)
        goal_rgb = T.ToTensor()(goal_rgb.convert("RGB"))
        goal_rgb = self.preprocess(goal_rgb) # (3, 224, 224)

        rgb = Image.fromarray(rgb)
        rgb = T.ToTensor()(rgb.convert("RGB"))
        rgb = self.preprocess(rgb) # (3, 224, 224)
        self.rgb_list.append(rgb)

        hand_rgb = Image.fromarray(hand_rgb)
        hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
        hand_rgb = self.preprocess(hand_rgb)
        self.hand_rgb_list.append(hand_rgb)

        state = obs['robot_obs'] # (15,)
        xyz_state = state[:3]
        rpy_state = state[3:6]
        rotm_state = euler2rotm(rpy_state)
        gripper_state = state[-1]
        state = (xyz_state, rotm_state, gripper_state)
        self.state_list.append(state)

        buffer_len = len(self.rgb_list)
        if buffer_len > self.seq_len:
            self.rgb_list.pop(0)
            self.hand_rgb_list.pop(0)
            self.state_list.pop(0)
            assert len(self.rgb_list) == self.seq_len
            assert len(self.hand_rgb_list) == self.seq_len
            assert len(self.state_list) == self.seq_len
            buffer_len = len(self.rgb_list)
        


        # Static RGB
        c, h, w = rgb.shape
        c2,h2,w2=goal_rgb.shape
        assert c==c2 and h==h2 and w==w2
        rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_tensor = torch.stack(self.rgb_list, dim=0) # (len, c, h, w)
        rgb_data[0, :buffer_len] = rgb_tensor
        goal_rgb_data=torch.zeros((1, c, h, w))
        goal_rgb_data[0]=goal_rgb

        # Hand RGB
        c, h, w = hand_rgb.shape
        hand_rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        hand_rgb_tensor = torch.stack(self.hand_rgb_list, dim=0) # (len, c, h, w)
        hand_rgb_data[0, :buffer_len] = hand_rgb_tensor

        # State
        arm_state, gripper_state = CustomModel.compute_rel_state(self.state_list)
        arm_state_data = torch.zeros((1, self.seq_len, 6))
        arm_state_tensor = torch.from_numpy(arm_state)
        arm_state_data[0, :buffer_len] = arm_state_tensor
        gripper_state_tensor = torch.from_numpy(gripper_state)
        gripper_state_tensor = (gripper_state_tensor + 1.0) / 2.0
        gripper_state_tensor = gripper_state_tensor.long()
        gripper_state_data = torch.zeros((1, self.seq_len)).long()
        gripper_state_data[0, :buffer_len] = gripper_state_tensor
        gripper_state_data = F.one_hot(gripper_state_data, num_classes=2).type_as(arm_state_data)

        # Attention mask
        attention_mask = torch.zeros(1, self.seq_len).long()
        attention_mask[0, :buffer_len] = 1

        # Action placeholder
        arm_action_data = torch.zeros((1, self.seq_len, self.configs["policy"]['act_len'], 6))
        gripper_action_data = torch.zeros(1, self.seq_len, self.configs["policy"]['act_len'])

        #progress_placeholder
        progress_data=torch.zeros(1, self.seq_len)

        input_dict = dict()
        input_dict['rgb'] = rgb_data.to(self.device)
        input_dict['hand_rgb'] = hand_rgb_data.to(self.device)
        input_dict["goal_rgb"]=goal_rgb_data.to(self.device)
        input_dict['arm_state'] = arm_state_data.to(self.device)
        input_dict['gripper_state'] = gripper_state_data.to(self.device)
        input_dict['arm_action'] = arm_action_data.to(self.device)
        input_dict['gripper_action'] = gripper_action_data.to(self.device)
        input_dict['attention_mask'] = attention_mask.to(self.device)
        input_dict["text"]=[text]
        input_dict["progress"]=progress_data
        # Forward pass
        with torch.no_grad():
            # action,action_traj = self.policy.evaluate(input_dict)
            action,progress = self.policy.evaluate(input_dict,return_progress=True)
            progress=int(int(progress * 10) *10)
            action=action.numpy()


        
        # Action mode: ee_rel_pose_local
        state = obs['robot_obs'] # (15,)
        xyz_state = state[:3]
        rpy_state = state[3:6]
        rotm_state = euler2rotm(rpy_state)
        rel_action = action
        xyz_action = rel_action[:3] / 50 # scale down by 50  
        rpy_action = rel_action[3:6] / 20 # scale down by 20
        gripper_action = rel_action[6]
        rotm_action = euler2rotm(rpy_action)
        xyz_next_state = xyz_state + rotm_state @ xyz_action
        rotm_next_state = rotm_state @ rotm_action
        rpy_next_state = rotm2euler(rotm_next_state)
        action = np.zeros(7)
        action[:3] = (xyz_next_state - xyz_state) * 50  
        action[3:6] = (rpy_next_state - rpy_state) * 20
        action[-1] = gripper_action
        action = torch.from_numpy(action)
        self.rollout_step_counter += 1
    
        return action,progress
