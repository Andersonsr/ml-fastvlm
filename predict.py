#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import os, sys
import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.encoder import lora
from util import load_safetensors_file
from llava.train.train_rx import find_all_linear_names
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def caption(model, tokenizer, image_processor, args, image_path):
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], args.prompt)
    conv.append_message(conv.roles[1], '')
    prompt = conv.get_prompt()
    # print(prompt)
    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        torch.device("cuda:0"))

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    # print(image_tensor.shape)
    # print(len(tokenizer))
    # print('input_ids length:', input_ids.shape[1])

    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=256,
            use_cache=True)

        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def predict(args):
    # Remove generation config from model folder
    # to read generation parameters from args
    # model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(args.model_path, 'generation_config.json')):
        generation_config = os.path.join(args.model_path, '.generation_config.json')
        os.rename(os.path.join(args.model_path, 'generation_config.json'),
                  generation_config)

    # Load model
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, 'llava-fastvithd_0.5b_stage3',
                                                             device="cuda:0")

    result = {'generated': []}
    root, extension = os.path.splitext(args.annotation)

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.to('cuda:0')
    # print(model.model.vision_tower.base_model.model)

    if extension.lower() == '.json':
        data = json.load(open(args.annotation, 'r'))
        for sample in tqdm(data):
            path = os.path.join(args.image_root, sample['image_name'])
            output = caption(model, tokenizer, image_processor, args, path)
            result['generated'].append({'id': sample['id'], 'prediction': output, 'reference': sample['findings']})

        json.dump(result, open(os.path.join(args.model_path, 'mimic-predictions.json'), 'w'), indent=2)

    elif extension.lower() == '.jpg' or extension.lower() == '.png':
        print(caption(model, tokenizer, image_processor, args, args.annotation))

    # Restore generation config
    if generation_config is not None:
        os.rename(generation_config, os.path.join(args.model_path, 'generation_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    # parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None, help="location of image file")
    parser.add_argument("--annotation", type=str, default=None, help="location of json annotation or image file")
    parser.add_argument("--conv_mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="describe the findings in <image>")
    args = parser.parse_args()

    predict(args)

