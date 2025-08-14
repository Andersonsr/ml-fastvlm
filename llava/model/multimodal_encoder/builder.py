import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .mobileclip_encoder import MobileCLIPVisionTower
from peft import LoraConfig, get_peft_model
import json


def build_vision_tower(vision_tower_cfg, **kwargs):
    # print(vision_tower_cfg)
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "mobileclip" in vision_tower.lower():
        path = vision_tower_cfg.model_name_or_path if hasattr(vision_tower_cfg, 'model_name_or_path') else vision_tower_cfg._name_or_path
        if os.path.exists(os.path.join(path, 'model_args.json')):
            config = json.load(open(os.path.join(path, 'model_args.json'), 'r'))

            if config['encoder_lora_enable']:
                model = MobileCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
                lora_config = LoraConfig(
                    r=config['encoder_lora_r'],
                    lora_alpha=config['encoder_lora_alpha'],
                    target_modules=['qkv'],
                    lora_dropout=config['encoder_lora_dropout'],
                    bias='lora_only',
                    task_type="IMAGE_CLASSIFICATION",
                )
                model.vision_tower = get_peft_model(model.vision_tower, lora_config)
                return model

        return MobileCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
