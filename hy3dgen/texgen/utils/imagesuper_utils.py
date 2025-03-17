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
from PIL import Image
import numpy as np

class Image_Super_Net():
    def __init__(self, config):
        # Configurações do modelo
        self.scale = 4  # Fator de upscaling (4x)
        model_path = "./RealESRGAN_x4plus_anime_6B.pth"

        # Carrega o modelo Real-ESRGAN
        model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=6,
            num_grow_ch=32,
            scale=self.scale
        )

        state_dict = torch.load(model_path, map_location=torch.device('cuda'))['params_ema']

        model.load_state_dict(state_dict, strict=True)

        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=model_path,
            model=model,
            tile=0,
            pre_pad=0,
            half=True
        )

    def __call__(self, image, prompt=''):  # Ignora o prompt (não usado no Real-ESRGAN)
        # Converte PIL.Image para numpy array
        img_array = np.array(image)
        
        # Realiza o upscaling
        upscaled_array, _ = self.upsampler.enhance(
            img_array, 
            outscale=self.scale
        )
        
        # Converte de volta para PIL.Image
        upscaled_image = Image.fromarray(upscaled_array)

        return upscaled_image

