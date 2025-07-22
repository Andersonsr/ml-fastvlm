import os
import sys
from PIL import Image
import torch

path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..', 'ml-fastvlm'))
sys.path.append(path)
# print(path)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.multimodal_encoder.mobileclip_encoder import MobileCLIPVisionTower
from peft import LoraConfig, get_peft_model
from llava.model.multimodal_encoder.mobileclip import MCi
from peft.peft_model import PeftModel


def get_encoder(model_path, dim):
    model_name = get_model_name_from_path(model_path)
    _, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, device="cuda:0", )
    if dim == 3072*256:
        return model.get_vision_tower(), image_processor

    elif dim == 768:
        return model.get_vision_tower().vision_tower, image_processor


def lora(model, r, alpha, dropout):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=['qkv'],
        lora_dropout=dropout,
        bias='lora_only',
        task_type="IMAGE_CLASSIFICATION",
    )
    return get_peft_model(model, lora_config)


def unfreeze_stages(model, modules):
    if type(model) == MobileCLIPVisionTower:
        network = model.vision_tower.model.network

    elif type(model) == MCi:
        network = model.model.network

    elif type(model) == PeftModel:
        if type(model.base_model.model) == MobileCLIPVisionTower:
            network = model.base_model.model.vision_tower.model.network

        elif type(model.base_model.model) == MCi:
            network = model.base_model.model.model.network

    else:
        raise NotImplementedError('Unsupported model type {}'.format(type(model)))

    for name, param in network[2].named_parameters():
        # if 'fc1' in name or 'fc2' in name:
        for module in modules:
            if module in name:
                param.requires_grad = True

    for name, param in network[4].named_parameters():
        # if 'fc1' in name or 'fc2' in name:
        for module in modules:
            if module in name:
                # print(name)
                param.requires_grad = True

    # print(model)

