import argparse
import logging
import os
import pickle
from tqdm import tqdm
import torch
import sys
import json
import numpy as np
from mimic_dataset import MimicDataset
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from model.encoder import get_encoder, lora


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='dir containing images')
    parser.add_argument('--output', type=str, required=True, help='dir to save embeddings chunks')
    parser.add_argument('--model', type=str, default=None, help='encoder model used to extract features',)
    parser.add_argument('--base_model', type=str, default='./checkpoints/llava-fastvithd_0.5b_stage3',
                        help='base model path')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--annotation', type=str, required=True, help='annotation file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    model, preprocess = get_encoder(args.base_model, 768, bf16=False)

    if args.model is not None:
        experiment = json.load(open(os.path.join(os.path.dirname(args.model), 'experiment.json'), 'r'))
        if experiment['lora']:
            model = lora(model, experiment['lora_rank'], experiment['lora_alpha'], experiment['lora_dropout'])
        model.load_state_dict(torch.load(args.model)['model_state_dict'])
        model.to(device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'text_embeddings': [], 'captions': [], 'labels': []}

    json_data = MimicDataset(args.root_dir, args.annotation)
    loader = json_data.get_loader(args.batch_size)
    logging.debug('Loaded chunk len: {}'.format(len(json_data)))

    for batch in tqdm(loader):
        logging.debug('batch size: {}'.format(len(batch['id'])))
        with torch.no_grad():
            embeddings = model(batch['image'].to(device))
            data['image_embeddings'] += embeddings
            data['text_embeddings'] += embeddings
            logging.debug('batch image embeddings shape: {}'.format(embeddings.shape))

        data['image_name'] += [name+'.png' for name in batch['id']]
        data['image_id'] += batch['id']
        data['captions'] += batch['findings']
        data['labels'] += batch['labels']

    data['image_embeddings'] = torch.stack(data['image_embeddings'], dim=0).unsqueeze(dim=1)
    data['text_embeddings'] = torch.stack(data['text_embeddings'], dim=0).unsqueeze(dim=1)
    logging.debug('image embeddings shape: {}'.format(data['image_embeddings'].shape))

    with open(args.output, 'wb') as f:
        pickle.dump(data, f)



