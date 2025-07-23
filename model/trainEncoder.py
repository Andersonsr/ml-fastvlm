import argparse
import logging
import os
import numpy as np
import sys
import torch
import json
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
# path trick
path = os.path.normpath(os.path.join(os.path.abspath(__file__), '..', '..'))
sys.path.append(path)
from util import learnable_parameters, model_size, plot_curves, balance_weights
from model.classifiers import MultiClassifier, mimic_classifier_list
from data.mimic_dataset import MimicDataset
from llava.mm_utils import get_model_name_from_path, process_images
from encoder import get_encoder, lora, unfreeze_stages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train encoder')
    parser.add_argument('--encoder-path', type=str, required=True, help='path to encoder checkpoint')
    parser.add_argument('--output_classes', type=int, default=3, choices=[3, 4], help='number of classes')
    parser.add_argument('--annotation', type=str, required=True, help='training dataset')
    parser.add_argument('--root-dir', type=str, required=True, help='path to dataset image dir')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory to save model and logs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--logging_interval', type=int, default=None,
                        help='how many batches to wait before logging')
    parser.add_argument('--validation_interval', type=int, default=None,
                        help='how many batches to wait before validating')
    parser.add_argument('--debug', action='store_true', default=False, help='log debug messages')
    parser.add_argument('--lora', action='store_true', default=False, help='apply lora')
    parser.add_argument('--lora_rank', type=int, default=16, help='rank of lora')
    parser.add_argument('--lora_alpha', type=int, default=32, help='alpha of lora')
    parser.add_argument('--lora_dropout', type=float, default=0.5, help='dropout of lora')
    parser.add_argument('--dim', type=int, default=3072*256, help='dimension of encoder output')
    parser.add_argument('--unfreeze', action='store_true', default=False, help='unfreeze')
    parser.add_argument('--modules', type=str, default=None, choices=['fc', 'mixer'])

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logging.info('device: {}'.format(device))
    logging.info('Loading data...')

    train_data = MimicDataset(args.root_dir, args.annotation, zeroed=True if args.output_classes == 3 else False)
    val_data = MimicDataset(args.root_dir, args.annotation.replace('train', 'dev'), zeroed=True if args.output_classes == 3 else False)
    classifiers_names = mimic_classifier_list

    train_dataloader = train_data.get_loader(args.batch_size)
    val_dataloader = val_data.get_loader(args.batch_size)

    # class weights
    weights = balance_weights(args.annotation, mimic_classifier_list, args.output_classes)
    logging.debug('number of training batches: {}'.format(len(train_dataloader)))
    logging.debug('number of training examples: {}'.format(len(train_data)))
    logging.debug('number of validation batches: {}'.format(len(val_dataloader)))

    if args.validation_interval is None:
        args.validation_interval = len(train_dataloader)

    if args.logging_interval is None:
        args.logging_interval = len(train_dataloader)

    os.makedirs(args.output_dir, exist_ok=True)
    multi_classifier = MultiClassifier(mimic_classifier_list, args.dim, args.output_classes)
    multi_classifier.to(device)

    name = get_model_name_from_path(args.encoder_path)
    encoder, preprocess = get_encoder(args.encoder_path, args.dim)
    encoder.to(device, dtype=torch.float)
    if args.lora:
        encoder = lora(encoder, args.lora_rank, args.lora_alpha, args.lora_dropout)

    if args.unfreeze:
        if args.modules == 'fc':
            modules = ['fc1', 'fc2']
        elif args.modules == 'mixer':
            modules = ['token_mixer']
        else:
            raise ValueError('arg modules must be "fc" or "mixer"')

        unfreeze_stages(encoder, modules)

    logging.info('backbone size: {}'.format(model_size(encoder)))
    logging.info('backbone learnable params: {}'.format(learnable_parameters(encoder)))
    logging.info('adapter size: {}'.format(model_size(multi_classifier)))
    logging.info('adapter learnable params: {}'.format(learnable_parameters(multi_classifier)))

    optim = Adam(list(encoder.parameters()) + list(multi_classifier.parameters()), lr=args.lr)

    # for logging purposes
    training_loss = []
    validation_loss = []
    classifier_loss = {}

    for classifier in classifiers_names:
        classifier_loss[classifier] = []

    for epoch in range(args.epochs):
        step_training_loss = []
        # epoch loss per classifier
        step_classifier_loss = {}
        for name in classifiers_names:
            step_classifier_loss[name] = []

        # training loop
        for i, batch in tqdm(enumerate(train_dataloader), desc="Epoch {}".format(epoch), total=len(train_dataloader)):
            optim.zero_grad()
            # images = preprocess()
            embeddings = encoder(batch['image'].to(device))
            # print(embeddings.shape)
            if len(embeddings.shape) > 2:
                b, c, d = embeddings.shape
                embeddings = embeddings.reshape(b, c*d)

            logging.debug('image shape: {}'.format(batch['image'].shape))
            logging.debug('embedding shape: {}'.format(embeddings.shape))

            classifier_logits = multi_classifier(embeddings)
            total_loss = []

            for name in classifiers_names:
                target = torch.tensor(batch['labels'][name], dtype=torch.long, device=device)
                CE = nn.CrossEntropyLoss(weight=weights[name].to(device, dtype=torch.float))
                loss = CE(classifier_logits[name], target)
                # print(loss)
                if not np.isnan(loss.cpu().detach().numpy()):
                    total_loss.append(loss)
                    # logging
                    step_classifier_loss[name].append(loss.detach().cpu().item())
                    classifier_loss[name].append(loss.detach().cpu().item())

                else:
                    step_classifier_loss[name].append(0.)
                    classifier_loss[name].append(0.)

            if len(total_loss) > 0:
                epoch_loss = sum(total_loss) / len(mimic_classifier_list)
                # print('loss', epoch_loss.item())
                epoch_loss.backward()
                optim.step()
                # logging
                step_training_loss.append(epoch_loss.detach().cpu().item())
            else:
                logging.warning(f'step {i} loss is nan')
                step_training_loss.append(0)

            # validation loop
            if (i+1) % args.validation_interval == 0 or i+1 == len(train_dataloader):
                step_validation_loss = []

                for batch in val_dataloader:
                    with torch.no_grad():
                        embeddings = encoder(batch['image'].to(device))
                        if len(embeddings.shape) > 2:
                            b, c, d = embeddings.shape
                            embeddings = embeddings.reshape(b, c * d)

                        classifier_logits = multi_classifier(embeddings)
                        total_loss = 0

                        for name in classifiers_names:
                            target = torch.tensor(batch['labels'][name], dtype=torch.long, device=device)
                            CE = nn.CrossEntropyLoss(weight=weights[name].to(device, dtype=torch.float))
                            loss = CE(classifier_logits[name], target)
                            if np.isnan(loss.cpu().detach().numpy()):
                                # all labels are equal to ignore index
                                loss = torch.tensor(0.0).to(device)
                            total_loss += loss
                        step_validation_loss.append(total_loss.item())

                validation_loss.append(sum(step_validation_loss)/len(step_validation_loss))

            # logging losses
            if (i+1) % args.logging_interval == 0 or i+1 == len(train_dataloader):
                training_loss.append(sum(step_training_loss) / len(step_training_loss))
                step_training_loss = []
                log = {'training_loss': training_loss, 'validation_loss': validation_loss}

                for classifier in classifiers_names:
                    classifier_loss[classifier].append(
                        sum(step_classifier_loss[classifier]) / len(step_classifier_loss[classifier]))
                    log[classifier + '_loss'] = classifier_loss[classifier]

                with open(os.path.join(args.output_dir, 'loss_log.json'), 'w') as f:
                    json.dump(log, f, indent=4)

                # plot graph with training and validation loss
                plot_curves(training_loss, validation_loss, os.path.join(args.output_dir, 'loss_plot.png'))

                # model checkpointing
                model_dict = {'epoch': epoch,
                              'model_state_dict': encoder.state_dict(),
                              'optimizer_state_dict': optim.state_dict(),
                              'loss': training_loss[-1]}

                torch.save(model_dict, os.path.join(args.output_dir, 'backbone_checkpoint.pt'))

                model_dict = {'epoch': epoch,
                              'model_state_dict': multi_classifier.state_dict(),
                              'optimizer_state_dict': optim.state_dict(),
                              'loss': training_loss[-1]}

                torch.save(model_dict, os.path.join(args.output_dir, 'classifier_checkpoint.pt'))

    # finito
    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.output_dir, 'backbone_checkpoint.pt')
    result_dict['classifiers_checkpoint'] = os.path.join(args.output_dir, 'classifier_checkpoint.pt')

    with open(os.path.join(args.output_dir, 'experiment.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)


