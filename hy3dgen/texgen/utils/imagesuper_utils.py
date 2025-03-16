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
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image

class Image_Super_Net():
    def __init__(self, config):
        # Usando ControlNet público para upscaling
        controlnet_model = "diffusers/controlnet-depth-sdxl-1.0"
        
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        
        self.pipe.set_progress_bar_config(disable=False)

    def __call__(self, image, prompt=''):
        # Pré-processamento adaptado para SDXL
        w, h = image.size
        control_image = image.resize((w*4, h*4), Image.Resampling.LANCZOS)
        
        with torch.no_grad():
            upscaled_image = self.pipe(
                prompt=f"high quality, detailed, {prompt}",
                negative_prompt="blurry, low quality, artifacts",
                image=control_image,
                num_inference_steps=30,
                controlnet_conditioning_scale=0.65,
                guidance_scale=7.5
            ).images[0]
            
        return upscaled_image

# import torch
# from diffusers import StableDiffusionUpscalePipeline
# #from PIL import ImageFilter

# class Image_Super_Net():
#     def __init__(self, config):
#         self.up_pipeline_x4 = StableDiffusionUpscalePipeline.from_pretrained(
#                         'stabilityai/stable-diffusion-x4-upscaler',
#                         variant="fp16",
#                         torch_dtype=torch.float16,
#                     ).to("cuda") # to(config.device)
#         self.up_pipeline_x4.set_progress_bar_config(disable=False)

#     def __call__(self, image, prompt=''):
#         with torch.no_grad():
#             upscaled_image = self.up_pipeline_x4(
#                 prompt="3D",
#                 image=image,
#                 num_inference_steps=20,
#             ).images[0]
#             #upscaled_image = upscaled_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

#         return upscaled_image