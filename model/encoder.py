import os
import sys
from PIL import Image
import torch

path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..', 'ml-fastvlm'))
sys.path.append(path)
# print(path)
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from torch import nn
from llava.model.multimodal_encoder.mobileclip_encoder import MobileCLIPVisionTower
from peft import LoraConfig, get_peft_model
from llava.model.multimodal_encoder.mobileclip import MCi
from util import learnable_parameters
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


def unfreeze_stages(model):
    if type(model) == MobileCLIPVisionTower:
        network = model.vision_tower.model.network

    elif type(model) == MCi:
        network = model.model.network

    elif type(model) == PeftModel:
        # print(type(model.base_model.model))
        if type(model.base_model.model) == MobileCLIPVisionTower:
            network = model.base_model.model.vision_tower.model.network

        elif type(model.base_model.model) == MCi:
            network = model.base_model.model.network

    else:
        raise NotImplementedError('Unsupported model type {}'.format(type(model)))

    for name, param in network[2].named_parameters():
        if 'fc1' in name or 'fc2' in name:
            # print(name)
            param.requires_grad = True

    for name, param in network[4].named_parameters():
        if 'fc1' in name or 'fc2' in name:
            # print(name)
            param.requires_grad = True


if __name__ == '__main__':
    model_path = '../checkpoints/llava-fastvithd_0.5b_stage3'
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,
                                                                           device="cuda:0")
    image = Image.open("E:\\datasets\\mimic\\preprocess\\resize_1024\\0000c2f5-f02f9f3c-1ed14642-958de0ad-d6ce4d20.jpg").convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0)
    print('image shape', image_tensor.shape)
    encoder = model.get_vision_tower().to(device="cuda:0", dtype=torch.float)
    output = encoder(image_tensor.to(device="cuda:0", dtype=torch.float))
    print('embeddings shape', output.shape)
    projector = model.model.mm_projector.to(dtype=torch.float32)
    projected = projector(output)
    print('projected shape', projected.shape)

    # print(model)

    # unfreeze_stages(model.get_vision_tower().vision_tower)

    # encoder = lora(model.get_vision_tower(), 16, 32, 0.5)
    # output = encoder(image_tensor)
    # print(learnable_parameters(encoder))


