import argparse
import json
import sys
import torch
import os
from sklearn.metrics import classification_report
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from data.mimic_dataset import MimicDataset
from model.classifiers import MultiClassifier
from tqdm import tqdm
from model.classifiers import mimic_classifier_list
from model.encoder import get_encoder, lora


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate image classification')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--root_dir', type=str, required=True, help='root directory containing the images')
    parser.add_argument('--annotation_file', type=str, required=True, help='annotation file')
    parser.add_argument('--model_path', type=str, required=True, help='path to saved model weights')
    parser.add_argument('--model_base_path', type=str, required=True, help='path to base model')
    parser.add_argument('--dim', type=int, default=256*3072)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "")
    data = MimicDataset(args.root_dir, args.annotation_file)
    loader = data.get_loader(args.batch_size)
    config = json.load(open(os.path.join(args.model_path, 'experiment.json'), 'r'))

    classifier = MultiClassifier(mimic_classifier_list, args.dim, config['output_classes']).to(device, dtype=torch.float)
    encoder, preprocess = get_encoder(args.model_base_path, args.dim)
    encoder.to(dtype=torch.float)

    if config['lora']:
        print('Loading lora model')
        encoder = lora(encoder, config['lora_rank'], config['lora_alpha'], config['lora_dropout'])
        model_dict = torch.load(os.path.join(args.model_path, 'backbone_checkpoint.pt'))['model_state_dict']
        encoder.load_state_dict(model_dict)

    else:
        print('loading model')
        model_dict = torch.load(os.path.join(args.model_path, 'backbone_checkpoint.pt'))['model_state_dict']
        encoder.load_state_dict(model_dict)

    model_dict = torch.load(os.path.join(args.model_path, 'classifier_checkpoint.pt'))['model_state_dict']
    classifier.load_state_dict(model_dict)

    predictions = {}
    gt = {}
    for name in mimic_classifier_list:
        predictions[name] = []
        gt[name] = []

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            image_features = encoder(batch['image'].to(device, dtype=torch.float))
            # b, c, d = image_features.shape
            # image_features = image_features.reshape(b, c * d)
            # print(classifier)
            classifier_logits = classifier(image_features)
            labels = batch['labels']
            for name in mimic_classifier_list:
                logits = classifier_logits[name]
                pred = torch.argmax(logits, dim=1).tolist()
                # print('a', pred)
                for i in labels[name]:
                    if i < config['output_classes']:
                        gt[name].append(labels[name][int(i)])
                        predictions[name].append(pred[int(i)])

    target_names = ['negative', 'positive', 'uncertain', 'not present']
    result_dict = []
    for name in mimic_classifier_list:
        report = classification_report(gt[name], predictions[name], labels=range(len(target_names)), target_names=target_names, zero_division=0, output_dict=True)
        result = {}
        # print(report)
        result['condition'] = name
        result['report'] = report
        # result['accuracy'] = report['accuracy']
        # result['macro_precision'] = report['macro avg']['precision']
        # result['macro_recall'] = report['macro avg']['recall']
        # result['macro_f1'] = report['macro avg']['f1-score']
        result_dict.append(result)

    output_file = os.path.join(args.model_path, 'classification_eval.json')
    json.dump(result_dict, open(output_file, 'w'), indent=2)

