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

# Caminho do arquivo (altere conforme o resultado do comando acima)
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
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from diffusers import StableDiffusionUpscalePipeline

from PIL import Image
import numpy as np
    
class Image_Super_Net():
    def __init__(self, config):
        
        # Carrega o modelo Real-ESRGAN
        # Configurações do modelo
        self.scale = 2  # Fator de upscaling (4x)
        self.tile_size = 0 # 512  # Processa a imagem em blocos para economizar VRAM
        self.tile_pad = 10
        #self.device = "cuda" #config.device  # Assume que config.device é "cuda" ou "cpu"

        model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23
        )

        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            model=model,
            tile=self.tile_size,
            tile_pad=self.tile_pad,
            device="cuda",
            pre_pad=10
        )
        
        # StableDiffusionUpscalePipeline
        self.up_pipeline_x2 = StableDiffusionUpscalePipeline.from_pretrained(
                        'stabilityai/sd-x2-latent-upscaler', #'stabilityai/stable-diffusion-x4-upscaler',
                        variant="fp16",
                        torch_dtype=torch.float16,
                    ).to("cuda") # to(config.device
        
        self.up_pipeline_x4.set_progress_bar_config(disable=False)

    def __call__(self, image, prompt=''):
        with torch.no_grad():
            
            # Inferencia com Real-ESRGAN
            img_array = np.array(image)
            upscaled_array, _ = self.upsampler.enhance(
                img_array, 
                outscale=self.scale
            )
            upscaled_image = Image.fromarray(upscaled_array)
            
            
            # Inferencia com StableDiffusionUpscalePipeline
            upscaled_image = self.up_pipeline_x2(
                prompt="high quality, detailed",
                negative_prompt="blurry, low quality, artifacts",
                image=upscaled_image,
                guidance_scale=0,
                num_inference_steps=5,
            ).images[0]

        return upscaled_image
    
import codecs

