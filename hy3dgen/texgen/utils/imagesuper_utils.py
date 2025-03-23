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
def adjust_rgb_to_grayscale(py_version):
    #print('Finding /usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py to update torchvision.transforms.functional_tensor location...')
    file_path = f"/usr/local/lib/python3.{py_version}/dist-packages/basicsr/data/degradations.py"
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
    
try:
    adjust_rgb_to_grayscale(10)
except Exception as e:
    pass 
try:
    adjust_rgb_to_grayscale(11)
except Exception as e:
    pass
        

import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Union

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


from diffusers.utils import load_image
from controlnet_aux import (
    MidasDetector,
    LineartDetector,
    PidiNetDetector,
    CannyDetector,
    NormalBaeDetector
)
from diffusers import (
    StableDiffusionXLControlNetUnionImg2ImgPipeline,
    ControlNetUnionModel,
    AutoencoderKL,
    DDIMScheduler
)


class Image_Super_Net():
    def __init__(self, config):
        
        self.device = config.device
        
        # creating_controlnet_sdxl_ip_pipeline
        controlnet_model = (
        ControlNetUnionModel
            .from_pretrained(
                "xinsir/controlnet-union-sdxl-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        )

        scheduler = (
            DDIMScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="scheduler",
                torch_dtype=torch.float16
            )
        )

        vae = (
            AutoencoderKL
            .from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16
            )
        )

        controlnet_model = (
            ControlNetUnionModel
            .from_pretrained(
                "xinsir/controlnet-union-sdxl-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        )

        pipe = (
            StableDiffusionXLControlNetUnionImg2ImgPipeline
            .from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet_model,
                scheduler=scheduler,
                vae=vae,
                variant="fp16",
                torch_dtype=torch.float16,
            )
        )

        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")
        self.pipe = pipe
        
        # processing_img
        self.processor_midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to('cuda')

        self.processor_lineart = LineartDetector.from_pretrained("lllyasviel/Annotators").to('cuda')

        self.processor_pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to('cuda')

        self.processor_normal = NormalBaeDetector.from_pretrained("lllyasviel/Annotators").to('cuda')
        
        self.processor_canny = CannyDetector()
        
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
        
        # Display images
        
        def exibir_imagens_lado_a_lado(imagens: Union[List, tuple], titulos: List[str] = None, figsize=(10, 10)):
            """
            Exibe uma lista de imagens lado a lado utilizando matplotlib.

            Parâmetros:
            - imagens: Lista ou tupla de imagens (PIL.Image ou arrays).
            - titulos: Lista de títulos para cada imagem. Se None, usa títulos vazios.
            - figsize: Tamanho da figura (largura, altura).
            """
            n = len(imagens)
            titulos = titulos if titulos is not None else [""] * n
            
            # Garante que a lista de títulos tenha o mesmo número de elementos que as imagens
            if len(titulos) < n:
                titulos += [""] * (n - len(titulos))
            elif len(titulos) > n:
                titulos = titulos[:n]

            fig, axs = plt.subplots(1, n, figsize=figsize)
            
            # Caso especial para 1 imagem (axs não é uma lista)
            if n == 1:
                axs = [axs]

            for ax, img, titulo in zip(axs, imagens, titulos):
                ax.imshow(img)
                ax.set_title(titulo)
                ax.axis('off')

            plt.tight_layout()
            plt.show()

    def __call__(self, view, image, prompt=''):
        
        size = 512
        
        view = view.resize((int(size), int(size)))
        image = image.resize((int(size), int(size)))
        
        controlnet_img = view

        controlnet_img_depth = self.processor_midas(controlnet_img).resize((int(size), int(size))).to("cuda")

        controlnet_img_lineart = self.processor_lineart(controlnet_img, resolution=size).resize((size, size)).to("cuda")

        controlnet_img_pidi = self.processor_pidi(controlnet_img).resize((size, size)).to("cuda")

        controlnet_img_normal = self.processor_normal(controlnet_img).resize((size, size)).to("cuda")
        
        controlnet_img_edges = self.processor_canny(controlnet_img, low_threshold=130, high_threshold=150).resize((size, size)).to("cuda")
        
        # 0 -- openpose
        # 1 -- depth
        # 2 -- hed/pidi/scribble/ted
        # 3 -- canny/lineart/anime_lineart/mlsd
        # 4 -- normal
        # 5 -- segment
        # 6 -- tile
        # 7 -- repaint
        control_images = [
            controlnet_img_depth,
            controlnet_img_pidi,
            controlnet_img_lineart,
            controlnet_img_edges,
            controlnet_img_normal
        ]
        control_mode = [
            1, 
            2, 
            3,
            3,
            4
        ]
        
        strength = 1.0
        guidance_scale = 5
        num_inference_steps = 40
        controlnet_conditioning_scale = 0.775
        ip_adapter_scale = 1.5
        
        prompt = """
        Ultra HD, 8k PBR texture, consistent material shading,
        detailed surface normals, photorealistic game asset,
        cinematic lighting, subsurface scattering
        """

        negative_prompt = """
        lowres, blurry, inconsistent texture, flat shading,
        oversaturated, unnatural lighting, plastic look
        """
        
        output_img= self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=view,
            ip_adapter_image=image,
            control_image=control_images,
            control_mode=control_mode,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
            width=image.width,
            height=image.height
        ).images[0]
        
        self.exibir_imagens_lado_a_lado([image, view, output_img])
        
        # Converte PIL.Image para numpy array
        img_array = np.array(output_img)
        
        # Realiza o upscaling
        upscaled_array, _ = self.upsampler.enhance(
            img_array, 
            outscale=self.scale
        )
        
        # Finalizando o upscaling
        upscaled_image = Image.fromarray(upscaled_array)

        return upscaled_image

