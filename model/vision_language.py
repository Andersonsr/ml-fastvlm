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

if __name__ == '__main__':
    model_path = '/src/ml-fastvlm/checkpoint\\llava-fastvithd_0.5b_stage3'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,
                                                                           device="cuda:0")
    image = Image.open("E:\\datasets\\mimic\\preprocess\\resize_1024\\0000c2f5-f02f9f3c-1ed14642-958de0ad-d6ce4d20.jpg").convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0)
    print(image_tensor.shape)
    output = model.get_vision_tower()(image_tensor)
    # print(output)
    projector = model.model.mm_projector.to(dtype=torch.float32)
    projected = projector(output)
    # print(projected.shape)
    # print(context_len)
    # print(conv_templates.keys())

    conv_qwen_2 = Conversation(
        system="<|im_start|>system\nYou are a medical AI assistant specialized in interpreting chest radiographs.",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        version="qwen_v2",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.QWEN_2,
        sep="<|im_end|>\n",
    )

    qs = 'describe the findings in <image>'
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates['qwen_2'].copy()

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)
    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(model.get_vision_tower().image_processor)
    # Tokenize prompt
    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("cuda:0"))
    # print(tokenizer)
