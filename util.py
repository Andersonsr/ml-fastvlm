import mmap

import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sklearn.utils.class_weight import compute_class_weight


def model_size(model):
    size_model = 0
    for param in model.parameters():
        if param.is_floating_point():
            size_model += param.numel() * torch.finfo(param.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.dtype).bits
    return f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB"


def learnable_parameters(model):
    learnable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            learnable += param.numel()

    return f'total params: {total / 1e6:.2f}M,  learnable params: {learnable / 1e6:.2f}M'


def plot_curves(training, validation, output_name):
    plt.plot(training, label=f'training loss')
    plt.text(len(training), training[-1], f'{training[-1]:.3}')

    if len(validation) > 0:
        val_interval = len(training)//len(validation)
        x = [i for i in range(val_interval-1, len(training), val_interval)]
        plt.plot(x, validation, label=f'validation loss')
        plt.text(x[-1], validation[-1], f'{validation[-1]:.3}')

    plt.title(f'training loss')
    plt.legend()
    plt.savefig(output_name)
    plt.clf()


# start of code from https://gist.github.com/Narsil/3edeec2669a5e94e4707aa0f901d2282
def load_safetensors_file(filename):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)

    size = os.stat(filename).st_size
    storage = torch.ByteStorage.from_file(filename, shared=False, size=size).untyped()
    offset = n + 8
    return {name: create_tensor(storage, info, offset) for name, info in metadata.items() if name != "__metadata__"}


DTYPES = {"F32": torch.float32, "I64": torch.int64, "BF16": torch.float16, }


def create_tensor(storage, info, offset):
    dtype = DTYPES[info["dtype"]]
    shape = info["shape"]
    start, stop = info["data_offsets"]
    return torch.asarray(storage[start + offset : stop + offset], dtype=torch.uint8).view(dtype=dtype).reshape(shape)

# end of code from  https://gist.github.com/Narsil/3edeec2669a5e94e4707aa0f901d2282


def balance_weights(json_file, class_list, number_of_classes):
    counts = {}
    for class_name in class_list:
        counts[class_name] = []

    for dado in json.load(open(json_file, 'r')):
        for label in class_list:
            if label in dado['labels'].keys():
                if dado['labels'][label] < number_of_classes:
                    counts[label].append(dado['labels'][label])

    weights = {}
    for class_name in class_list:
        occurrences = np.array(counts[class_name])
        weight = compute_class_weight('balanced', classes=np.unique(occurrences), y=counts[class_name])
        weights[class_name] = torch.tensor(weight)
    # print(weights)
    return weights


if __name__ == "__main__":
    # from model.classifiers import mimic_classifier_list
    # path = 'E:\\datasets\\mimic\\preprocess\\train_split.json'
    # balance_weights(path, mimic_classifier_list, 4)
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    model_path = 'checkpoints/llava-fastvithd_0.5b_stage3'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,
                                                                           device="cuda:0")

    state = load_safetensors_file('checkpoints/llava-fastvithd_0.5b_stage3/model.safetensors')
    weights = {'0.bias': state['model.mm_projector.0.bias'],
               '0.weight': state['model.mm_projector.0.weight'],
               '2.weight': state['model.mm_projector.2.weight'],
               '2.bias': state['model.mm_projector.2.bias']}
    model.model.mm_projector.load_state_dict(weights)

