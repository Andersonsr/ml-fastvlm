import argparse
import torch
import os
import sys
import logging
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import math
from peft import LoraConfig, get_peft_model
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from util import learnable_parameters
from model.mapper import create_mapper
from model.classifiers import mimic_classifier_list

logger = logging.getLogger('captioning')
# for key in os.environ.keys():
#     print(key, os.environ[key])


class Decoder(nn.Module):
    def __init__(self, model_name, device, precision=torch.float16, prefix_length=10, add_noise=False, variance=0.016,
                 input_dimension=768, normalize=False, prefix_before_bos=False, append_eos=False):
        super(Decoder, self).__init__()
        self.device = device
        self.before_bos = prefix_before_bos
        self.precision = precision

        if 'opt' in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=precision,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.ignore_id = -100

        elif 'llama' in model_name:
            assert 'HF_TOKEN' in os.environ.keys(), 'HF_TOKEN environment variable not set'
            # login(token=os.environ['HF_TOKEN'])
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=precision)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

            # <|eot_id|> = 128009, <|end_of_text|> = 128001
            self.tokenizer.pad_token_id = 128009
            self.model.generation_config.pad_token_id = 128009

            self.tokenizer.eos_token_id = 128001
            self.model.generation_config.eos_token_id = 128001
            self.ignore_id = -100

        else:
            raise ValueError(f'{model_name} not supported')

        self.append_eos = append_eos
        self.add_noise = add_noise
        self.variance = variance
        self.hidden_size = self._get_hidden_size()
        self.prefix_length = prefix_length
        self.fp = precision
        self.mapper = create_mapper(input_dimension, self.hidden_size, len(mimic_classifier_list)).to(dtype=precision)
        self.normalize = normalize

        logging.debug(f'hidden size: {self.hidden_size}')
        logging.debug(f'BOS token id: {self.tokenizer.bos_token_id}')
        logging.debug(f'EOS token: {self.tokenizer.eos_token}')
        logging.debug(f'EOS token id: {self.tokenizer.eos_token_id}')
        logging.debug(f'PAD token: {self.tokenizer.pad_token}')
        logging.debug(f'PAD token id: {self.tokenizer.pad_token_id}')

        if self.device:
            self.model.to(self.device)
            self.mapper.to(self.device)

    def caption(self, embeddings, do_sample=False, max_tokens=200, seed=32, num_beams=1, top_k=None, top_p=None,
                temperature=1.0, penalty_alpha=None, diversity_penalty=None):
        set_seed(seed)
        if self.normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        logging.debug(f'input embeddings shape:{embeddings.shape}')
        embeddings = embeddings.to(self.device)
        patches = 4 if len(embeddings.shape) > 2 else 1

        logging.debug(f'reshaped embeddings shape:{embeddings.shape}')
        prefix = self.mapper(embeddings.to(dtype=self.fp)).view(-1, patches*self.prefix_length, self.hidden_size)

        # reshape to 1, patches*maper_out, decoder_dim
        logging.debug(f'prefix shape: {prefix.shape}')

        # id do token de inicio de frase
        bos_token = torch.ones((1, 1)).to(dtype=torch.long) * self.tokenizer.bos_token_id
        bos_token = bos_token.to(self.device)
        embeddings_layer = self.model.get_input_embeddings()
        bos_embeddings = embeddings_layer(bos_token)


        if self.before_bos:
            prefix = torch.concat([prefix, bos_embeddings], dim=1)
        else:
            prefix = torch.concat([bos_embeddings, prefix], dim=1)

        attention_mask = torch.ones((prefix.shape[0], prefix.shape[1])).to(self.device, dtype=torch.long)
        # print('ATTENTION SHAPE', attention_mask.shape)

        logging.debug(f'decoder input shape: {prefix.shape}')
        attention_mask = torch.ones(prefix.shape[:2]).to(self.device, dtype=self.precision)
        generated_ids = self.model.generate(do_sample=do_sample,
                                            max_new_tokens=max_tokens,
                                            inputs_embeds=prefix,
                                            attention_mask=attention_mask,
                                            num_beams=num_beams,
                                            top_k=top_k,
                                            top_p=top_p,
                                            temperature=temperature,
                                            penalty_alpha=penalty_alpha,
                                            diversity_penalty=diversity_penalty)

        logging.debug(f'Generated ids: {generated_ids}')
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def forward(self, batch):
        embeddings = batch['embeddings'].to(dtype=self.fp)
        captions = batch['captions']
        logging.debug(f'input embeddings shape: {embeddings.shape}')
        if self.add_noise:
            embeddings = self.noise_injection(embeddings)
        if self.device:
            embeddings = embeddings.to(self.device)
        if self.normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        if self.append_eos:
            captions = [caption + self.tokenizer.eos_token for caption in captions]

        # batch size, patches, model dim
        b, p, d = embeddings.shape
        embeddings = embeddings.view(b*p, 1, d)

        prefix_tokens = self.mapper(embeddings).view(-1, self.prefix_length, self.hidden_size)
        prefix_tokens = prefix_tokens.view(b, p*self.prefix_length, self.hidden_size)

        logging.debug(f'Mapper output: {prefix_tokens.shape}')

        captions_emb = self.get_input_embeds(captions).to(dtype=self.fp, device=self.device)

        # print("bos ", captions_emb[:, :1, :].shape)
        logging.debug(f'captions embeddings shape: {captions_emb.shape}')

        if len(captions_emb.shape) == 2:
            captions_emb = captions_emb.unsqueeze(0)
            logging.debug(f' captions embeddings unsqueeze shape: {captions_emb.shape}')

        # final shape [batch, sos + prefix + caption len-1, d_model]
        if self.before_bos:
            input_emb = torch.concat([prefix_tokens, captions_emb], dim=1).to(self.fp)

        else:
            input_emb = torch.concat([captions_emb[:, :1, :], prefix_tokens, captions_emb[:, 1:, :]], dim=1).to(self.fp)

        # labels for auto regressive CE training
        labels = self.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(self.device, dtype=self.fp)

        logging.debug(f'concatenated embeddings final shape: {input_emb.shape}')
        logging.debug('labels shape: {}'.format(labels.shape))

        # print('LABELS', labels[0])
        # ignore padding tokens
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_id

        # ignore prefix, set labels to skip prefix during loss computation
        ignore_size = self.prefix_length*p
        ignore = torch.ones(input_emb.shape[0],  ignore_size) * self.ignore_id
        logging.debug('ignore shape: {}'.format(ignore.shape))

        labels = labels.to(self.device)
        ignore = ignore.to(self.device)
        input_emb = input_emb.to(self.device)
        # concatenate prefix labels (ignore) and text labels
        if self.before_bos:
            # prefix, bos, text
            labels = torch.concat([ignore, labels], dim=1)

        else:
            # bos, prefix, text
            labels = torch.concat([labels[:, :1], ignore, labels[:, 1:]], dim=1)

        logging.debug('final labels shape: {}'.format(labels.shape))
        return self.model(inputs_embeds=input_emb, labels=labels.to(torch.long))

    def get_input_embeds(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt", padding=True).input_ids.to(self.device).squeeze(0)
        embeddings_layer = self.model.get_input_embeddings()
        return embeddings_layer(input_ids)

    def _get_hidden_size(self):
        ids = self.tokenizer("prompt", return_tensors="pt").input_ids.squeeze(0)
        embeddings = self.model.get_input_embeddings()
        if self.device:
            ids = ids.to(self.device)
            embeddings = embeddings.to(self.device)

        return embeddings(ids).shape[1]

    def noise_injection(self, x, ):
        x = x.to('cuda')
        return x + torch.randn(x.shape, device='cuda', dtype=self.fp) * math.sqrt(self.variance)

    def lora_model(self, r, alpha, dropout):
        for param in self.model.parameters():
            param.requires_grad = False

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",

        )
        self.model = get_peft_model(self.model, config).to(self.fp)


def model_from_json(json_file, device):
    import json
    import os
    config = json.load(open(json_file, 'r'))

    precision = torch.float16 if config['fp'] == 'fp16' else torch.float32
    before_bos = config['before_bos'] if 'before_bos' in config else False
    append_eos = config['append_eos'] if 'append_eos' in config else False

    normalize = config['normalize'] if 'normalize' in config else False
    # do not add noise during eval
    decoder = Decoder(config['model_name'], device, prefix_length=config['prefix_len'], precision=precision,
                      add_noise=False, input_dimension=config['dimension'], prefix_before_bos=before_bos,
                      normalize=normalize, append_eos=append_eos)

    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    if not os.path.exists(config['model_name']) and not config['full_finetune']:
        # loaded model is on the hub and was not adapted before, need to create adapter
        decoder.lora_model(config['rank'], config['alpha'], config['dropout'])

    decoder.load_state_dict(checkpoint['model_state_dict'])

    print('loaded model from {}'.format(json_file))
    learnable_parameters(decoder)

    return decoder
