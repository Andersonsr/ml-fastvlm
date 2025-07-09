import json
import torch
from open_clip import create_model_from_pretrained
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import timm
import open_clip


def model_from_config(config):
    if config['package'] == 'timm':
        model = timm.create_model(config['model_name'], pretrained=True, num_classes=0)

    elif config['package'] == 'openclip':
        model, _ = create_model_from_pretrained(config['model_name'])
        model = model.visual

    elif config['package'] == 'transformers':
        model = AutoModel.from_pretrained(config['model_name'])

    else:
        raise ValueError(f'Unknown package: {config["package"]}')

    if 'lora' in config and config['lora']:
        # TODO: apply LoRA
        raise NotImplementedError

    return model


def create_model(config):
    # fine-tuned local model
    if 'encoder_config' in config.keys():
        model = model_from_config(config['encoder_config'])
        checkpoint = torch.load(config['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])

    else:
        model = model_from_config(config)

    return model


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.vision = create_model(config)
        if 'encoder_config' in config.keys():
            config = config['encoder_config']

        self.input_size = config['input_size']
        self.output_dim = config['hidden_size']
        self.package = config['package']
        self.lora = config['lora'] if 'lora' in config.keys() else False

    def forward(self, x):
        if self.package == 'timm' or self.package == 'openclip':
            return self.vision(x)

        elif self.package == 'transformers':
            return self.vision(x)['pooler_output']

    def grid_features(self, x):
        if self.package == 'timm':
            return self.vision.forward_features(x)

        if self.package == 'openclip':
            features = {}
            def hook(module, input, output):
                features['output'] = output

            self.vision.trunk.blocks[-1].register_forward_hook(hook)
            self.vision(x)
            return features['output']

        elif self.package == 'transformers':
            return self.vision(x)['last_hidden_state']


if __name__ == '__main__':
    siglip_512 = {'input_size': 512,
              'hidden_size': 768,
              'model_name': 'vit_base_patch16_siglip_512',
              'package': 'timm'}

    siglip2_256 = {'input_size': 256,
                   'hidden_size': 768,
                   'model_name': 'hf-hub:timm/ViT-B-16-SigLIP2-256',
                   'package': 'openclip'}

    dinov2 = {'input_size': 224,
              'hidden_size': 768,
              'model_name': 'facebook/dinov2-with-registers-base',
              'package': 'transformers'}

    cfg = json.load(open('D:\\modelos_v2\\encoder\\class3_siglip256\\experiment.json'))
    vision = Encoder(cfg)
    x = torch.randn(1, 3, 256, 256)
    print(vision(x).shape)

