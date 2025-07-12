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


def get_encoder(model_path, model_name):
    _, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, device="cuda:0")
    return model.model.vision_tower, image_processor


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


if __name__ == '__main__':
    model_path = '../checkpoints/llava-fastvithd_0.5b_stage3'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,
                                                                           device="cuda:0")
    image = Image.open("E:\\datasets\\mimic\\preprocess\\resize_1024\\0000c2f5-f02f9f3c-1ed14642-958de0ad-d6ce4d20.jpg").convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0)
    print('image shape', image_tensor.shape)
    output = model.get_vision_tower()(image_tensor)
    print('embeddings shape', output.shape)
    projector = model.model.mm_projector.to(dtype=torch.float32)
    projected = projector(output)
    print('projected shape', projected.shape)

    # for name, module in model.named_modules():
    #     print(name)
    print(model.get_vision_tower().vision_tower.model.network)
    # print(model.get_vision_tower().vision_tower.model.network[7])
    # encoder = lora(model.get_vision_tower(), 16, 32, 0.5)
