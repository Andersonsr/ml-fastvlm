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
path = os.path.normpath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
sys.path.append(path)
from util import learnable_parameters, model_size, plot_curves
from model.classifiers import MultiClassifier, mimic_classifier_list
from data.mimic_dataset import MimicDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train encoder')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--lora', default=False, action='store_true', help='apply lora')
    parser.add_argument('--output_classes', type=int, default=3, help='number of classes')
    parser.add_argument('--dataset', type=str, required=True, choices=['mimic'], help='training dataset')
    parser.add_argument('--data', type=str, required=True, help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory to save model and logs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--chunks', default=None, type=int, help='number of chunks to use')
    parser.add_argument('--logging_interval', type=int, default=None,
                        help='how many batches to wait before logging')
    parser.add_argument('--validation_interval', type=int, default=None,
                        help='how many batches to wait before validating')
    parser.add_argument('--debug', action='store_true', default=False, help='log debug messages')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    assert os.path.exists(args.config), 'config file not found: {}'.format(args.config)
    config = json.load(open(args.config, 'r'))
    logging.info('device: {}'.format(device))
    logging.info('Loading data...')
    if args.dataset == 'mimic':
        train_data = MimicDataset(args.data, chunks=args.chunks)
        val_data = MimicDataset(args.data.replace('train', 'dev'), )
        classifiers_names = mimic_classifier_list

    else:
        raise NotImplementedError

    train_dataloader = train_data.get_loader(args.batch_size)
    val_dataloader = val_data.get_loader(args.batch_size)

    logging.debug('number of training batches: {}'.format(len(train_dataloader)))
    logging.debug('number of training examples: {}'.format(len(train_data)))
    logging.debug('number of validation batches: {}'.format(len(val_dataloader)))

    if args.validation_interval is None:
        args.validation_interval = len(train_dataloader)

    if args.logging_interval is None:
        args.logging_interval = len(train_dataloader)

    os.makedirs(args.output_dir, exist_ok=True)
    multi_classifier = MultiClassifier(mimic_classifier_list, config['output_dim'], args.output_classes)

    encoder = Encoder(config)
    encoder.to(device)
    multi_classifier.to(device)

    logging.info('backbone size: {}'.format(model_size(encoder)))
    logging.info('backbone learnable params: {}'.format(learnable_parameters(encoder)))
    logging.info('adapter size: {}'.format(model_size(multi_classifier)))
    logging.info('adapter learnable params: {}'.format(learnable_parameters(multi_classifier)))

    optim = Adam(list(encoder.parameters()) + list(multi_classifier.parameters()), lr=args.lr)
    CE = nn.CrossEntropyLoss(ignore_index=args.output_classes)

    if args.lora:
        # TODO: apply LoRA here
        raise NotImplementedError

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
            embeddings = encoder(batch['image_tensor'].to(device))

            logging.debug('image shape: {}'.format(batch['image_tensor'].shape))
            logging.debug('embedding shape: {}'.format(embeddings.shape))

            classifier_logits = multi_classifier(embeddings)
            accumulated_loss = 0
            for name in classifiers_names:
                logging.debug('classifier logits {}: {}'.format(name, classifier_logits[name].shape))
                logging.debug('labels shape {}: {}'.format(name, batch['labels'][name].shape))
                loss = CE(classifier_logits[name], batch['labels'][name].to(device)) / len(classifiers_names)
                if np.isnan(loss.cpu().detach().numpy()):
                    # all labels are equal to ignore index
                    loss = torch.tensor(0.0).to(device)
                step_classifier_loss[name].append(loss.item())
                accumulated_loss += loss

            accumulated_loss.backward()
            step_training_loss.append(accumulated_loss.item())

            optim.step()
            optim.zero_grad()

            # validation loop
            if (i+1) % args.validation_interval == 0 or i+1 == len(train_dataloader):
                step_validation_loss = []

                for batch in val_dataloader:
                    with torch.no_grad():
                        embeddings = encoder(batch['image_tensor'].to(device))
                        classifier_logits = multi_classifier(embeddings)
                        accumulated_loss = 0

                        for name in classifiers_names:
                            loss = CE(classifier_logits[name], batch['labels'][name].to(device)) / len(
                                classifiers_names)
                            if np.isnan(loss.cpu().detach().numpy()):
                                # all labels are equal to ignore index
                                loss = torch.tensor(0.0).to(device)
                            accumulated_loss += loss
                        step_validation_loss.append(accumulated_loss.item())

                validation_loss.append(sum(step_validation_loss) / len(step_validation_loss))

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
                              'model_state_dict': encoder.vision.state_dict(),
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
    result_dict['encoder_config'] = config

    with open(os.path.join(args.output_dir, 'experiment.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)


