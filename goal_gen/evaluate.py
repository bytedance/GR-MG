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
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from goal_gen.utils.pipeline import Pipeline
from goal_gen.data.calvindataset import CalvinDataset_Goalgen
class IP2PEvaluation(object):
    def __init__(self, 
                 ckpt_path,
                 res=256):    
        # Init models
        pretrained_model_dir = "/mnt/bn/lpy-lq/stable_diffusion/instruct-pix2pix"
        
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_dir, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_dir, subfolder="unet")

        # Load weight for unet
        payload = torch.load(ckpt_path)
        state_dict = payload['state_dict']
        del payload
        msg = self.unet.load_state_dict(state_dict['unet_ema'], strict=True)
        print(msg)

        self.pipe = Pipeline.from_pretrained(
            pretrained_model_dir,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            vae=self.vae,
            unet=self.unet,
            revision=None,
            variant=None,
            torch_dtype=torch.bfloat16
        ).to("cuda")

        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.generator = torch.Generator("cuda").manual_seed(42)

        # Diffusion hyparams
        self.num_inference_steps = 50
        self.image_guidance_scale = 2.5
        self.guidance_scale = 7.5

        # Image transform
        self.res = res
        self.transform = transforms.Resize((res, res))

    def evaluate(self, eval_result_dir, eval_data_dir,is_training):
        os.makedirs(eval_result_dir,exist_ok=True)
        save_dir=os.path.join(eval_result_dir,"debug.png")
        dataset = CalvinDataset_Goalgen(
            eval_data_dir, 
            resolution=256,
            resolution_before_crop=288,
            center_crop=True,
            forward_n_min_max=(20, 22), 
            is_training=is_training,
            use_full=True,
            color_aug=False
        )
        for i in range(0, len(dataset), 100):
            example = dataset[i]
            text=example['input_text']
            original_pixel_values = example['original_pixel_values']
            edited_pixel_values = example['edited_pixel_values']
            progress=example["progress"]

            progress=progress*10
            text[0]=text[0]+f".And {progress}% of the instruction has been finished." 
            print(text[0])
            input_image_batch=[original_pixel_values]
            predict_image = self.inference(input_image_batch, text)

            fig, ax = plt.subplots(1,3)
            for k in range(3):
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

                ax[2].imshow(predict_image[0])

            plt.savefig( save_dir,dpi=300)
            plt.close()

    def inference(self, image_batch, text_batch):
        """Inference function."""
        input_images = []
        for image in image_batch:
            if isinstance(image, np.ndarray):
                image=Image.fromarray(image)
            input_image = self.transform(image)
            input_images.append(input_image)
        edited_images = self.pipe(
            prompt=text_batch,
            image=input_images,
            num_inference_steps=self.num_inference_steps,
            image_guidance_scale=self.image_guidance_scale,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
            safety_checker=None,
            requires_safety_checker=False).images
        edited_images=[ np.array(image) for image in edited_images]

        return edited_images

if __name__ == "__main__":  
    ckpt_path="PATH_TO_IP2P_CKPT/epoch=49-step=102900.ckpt"
    eval = IP2PEvaluation(ckpt_path)
    eval_data_dir = "PATH_TO_CALVIN/calvin/task_ABC_D/"
    eval_result_dir = "SAVE_DIR"
    eval.evaluate(eval_result_dir, eval_data_dir,is_training=False)