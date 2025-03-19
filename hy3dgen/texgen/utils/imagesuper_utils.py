# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler

from .imagesuper_filter import *

class Image_Super_Net():
    def __init__(self, config):
        
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=ControlNetModel.from_pretrained(
                "xinsir/controlnet-tile-sdxl-1.0",
                torch_dtype=torch.float16
            ),
            vae=AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", 
                torch_dtype=torch.float16
            ),
            scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                subfolder="scheduler"
            ),
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        
        # self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_vae_slicing()
        
        self.pipe

    def __call__(self, image, prompt=''): 
        # Converte PIL.Image para numpy array
        image = controlnet_img(image)
        
        controlnet_conditioning_scale = 1.0  
        prompt = "3D, highly detailed, sharp focus,  high textures, 4k, Game AAA"
        negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, missing face, extra digit, fewer digits, cropped, worst quality, low quality'

        upscaled_image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=1024,
            height=1024,
            num_inference_steps=50,
        ).images[0]

        return upscaled_image

