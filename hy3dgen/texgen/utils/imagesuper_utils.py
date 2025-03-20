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

import codecs

# Caminho do arquivo (altere conforme o resultado do comando acima)
# !find -name "degradations.py"
print('Finding /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py to update torchvision.transforms.functional_tensor location...')
file_path = "/usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py"
# Lê o conteúdo do arquivo
with codecs.open(file_path, "r", "utf-8") as f:
    content = f.read()

# Substitui a linha problemática
new_content = content.replace(
    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
    "from torchvision.transforms.functional import rgb_to_grayscale"
)

# Salva as alterações
with codecs.open(file_path, "w", "utf-8") as f:
    f.write(new_content)
print("Successfully!")

import torch
from PIL import Image
import numpy as np
import cv2
from IPython.display import display

from diffusers import ControlNetUnionModel, AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
from mod_controlnet_tile_sr_sdxl import StableDiffusionXLControlNetTileSRPipeline
from util import (
    create_hdr_effect,
    progressive_upscale,
    quantize_8bit,
    select_scheduler,
)

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class Image_Super_Net():
    def __init__(self, config):
        # Carrega o Stable Diffusion XL 1.0 Upscaling com ajustes de textura

        self.pipe = StableDiffusionXLControlNetTileSRPipeline.from_pretrained(
                "SG161222/RealVisXL_V5.0", 
            controlnet=ControlNetUnionModel.from_pretrained(
                "brad-twinkl/controlnet-union-sdxl-1.0-promax", 
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
            use_safetensors=True, 
            variant="fp16"
        ).to('cuda')

        # self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_vae_tiling()
        # self.pipe.enable_vae_slicing()
        
        # Carrega o modelo Real-ESRGAN
        self.scale = 2
        
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=2
        )
        
        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            model=model,
            tile=0,
            tile_pad=0,
            pre_pad=0,
            half=False,
            gpu_id='0'
        ) 

    def __call__(self, image, prompt=''):
        
        original_height = image.height
        original_width = image.width

        # Pre-upscale image for tiling
        resolution = 1024
        hdr = 0.5
        tile_gaussian_sigma = 0.3
        max_tile_size = 1024 # or 1280
        control_image = create_hdr_effect(image, hdr)
        image = progressive_upscale(image, resolution)
        image = create_hdr_effect(image, hdr)

        # Update target height and width
        target_height = resolution # image.height
        target_width = resolution # image.width
        
        # Calculate overlap size
        normal_tile_overlap, border_tile_overlap = self.pipe.calculate_overlap(target_width, target_height)
        
        tile_weighting_method = self.pipe.TileWeightingMethod.COSINE.value
        guidance_scale = 2.7
        num_inference_steps = 50
        denoising_strenght = 1.0
        controlnet_strength = 0.435
        prompt = "high-quality, noise-free edges, high quality"
        negative_prompt = "blurry, pixelated, noisy, low resolution, artifacts, poor details"

        # Image generation
        sd_image_output = self.pipe(
            image=control_image,
            control_image=image,
            control_mode=[6],
            controlnet_conditioning_scale=float(controlnet_strength),
            prompt=prompt,
            negative_prompt=negative_prompt,
            normal_tile_overlap=normal_tile_overlap,
            border_tile_overlap=border_tile_overlap,
            height=target_height,
            width=target_width,
            original_size=(original_width, original_height),
            target_size=(target_width, target_height),
            guidance_scale=guidance_scale,        
            strength=float(denoising_strenght),
            tile_weighting_method=tile_weighting_method,
            max_tile_size=max_tile_size,
            tile_gaussian_sigma=float(tile_gaussian_sigma),
            num_inference_steps=num_inference_steps,
        )["images"][0]
        display(sd_image_output)
        
        # Converte PIL.Image para numpy array
        img_array = np.array(sd_image_output)
        
        # Realiza o upscaling
        upscaled_array, _ = self.upsampler.enhance(
            img_array, 
            outscale=self.scale
        )
        
        # Finalizando o upscaling
        upscaled_image = Image.fromarray(upscaled_array)

        return upscaled_image

