import argparse
import pickle
import random
import logging
import torch
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
import json
import os
import sys
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from decoder import Decoder
from data.dataLoaders import COCODataset, PetroDataset, MIMICLoader
from util import learnable_parameters


def prepare_batch(batch, text_only, patch, device, num_descriptions=5, break_line=False):
    '''
    Prepare the batch to be forwarded to the model
    :param batch: batch to be processed
    :param text_only: to use text only or not (Boolean)
    :param patch: patch embeddings
    :param device: device to use for computation
    :param num_descriptions: total number of descriptions for each image, used to randomize the captioning
    :param break_line: break caption separated by new line character
    :return: object with keys 'caption' and 'embeddings'
    '''

    if text_only:
        embeds = batch['text_embeddings']
        logging.debug(f'prepare batch using text embeddings')

    else:
        if patch:
            embeds = batch['patch_embeddings']
        else:
            embeds = batch['image_embeddings']
            logging.debug(f'prepare batch using image embeddings')

    embeds = embeds.to(device)
    if num_descriptions > 1:
        # random description
        c = random.randint(0, num_descriptions-1)
        logging.debug(f'prepare batch randomized caption {c} of {num_descriptions-1}')
        captions = [caption[c] for caption in batch['captions']]

    else:
        # only one description
        logging.debug(f'prepare batch, only one caption')
        captions = batch['captions']
        # print(captions)

    if break_line:
        captions = [s.split('\n')[0] for s in captions]

    # print(len(captions), embeds.shape)
    return {'captions': captions, 'embeddings': embeds}


def train(epochs, batch_size, lr, filename, r, alpha, dropout, model_name, prefix_len, fp, text_only,
          full_finetune, add_noise, variance, save_history, dataset, root, dimension, log_step,
          normalize, patch, before, break_line, append_eos, pretrained_mapper):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('model device {}'.format(device))
    # data
    num_captions = 1
    if dataset == 'coco':
        train_data = COCODataset(filename, 5)
        val_name = filename.replace('train', 'val')
        val_data = COCODataset(val_name, 5)
        num_captions = 5

    elif dataset == 'petro':
        train_data = PetroDataset(filename, split='train')
        val_data = PetroDataset(filename, split='val')

    elif dataset == 'cego':
        train_data = PetroDataset(filename, None)
        val_name = filename.replace('train', 'val')
        val_data = PetroDataset(val_name, None)

    elif dataset == 'mimic':
        train_data = MIMICLoader(filename)
        val_data = MIMICLoader(filename.replace('train', 'dev'))

    else:
        raise ValueError(f'{dataset} is not a valid dataset')

    logging.debug('training dataset size: %d' % len(train_data))
    logging.debug('validation dataset size: %d' % len(val_data))

    train_loader = train_data.get_loader(batch_size=batch_size)
    val_loader = val_data.get_loader(batch_size=batch_size)

    # model
    decoder = Decoder(model_name, device,
                      prefix_length=prefix_len,
                      precision=fp,
                      add_noise=add_noise,
                      variance=variance,
                      input_dimension=dimension,
                      normalize=normalize,
                      prefix_before_bos=before,
                      append_eos=append_eos)

    if pretrained_mapper is not None:
        assert os.path.exists(pretrained_mapper), 'pretrained mapper not found at {}'.format(pretrained_mapper)
        decoder.mapper = torch.load(pretrained_mapper)['state_dict']

    if not full_finetune:
        # model was adapted before, load existing adapter to continue training
        if os.path.exists(os.path.join(model_name, 'adapter_config.json')):
            logging.debug('loaded existing adapter')
            decoder.model.enable_adapters()

        else:
            # create new adapter
            decoder.lora_model(r, alpha, dropout)
            logging.debug('created new adapter')

    optim = AdamW(decoder.parameters(), lr=lr)

    logging.debug('DECODER SIZE {}'.format(learnable_parameters(decoder.model)))
    logging.debug('MAPPER SIZE {}'.format(learnable_parameters(decoder.mapper)))

    training_losses = []
    validation_losses = []

    if log_step is None:
        log_step = len(train_loader)

    # training loop
    for epoch in range(epochs):
        log_loss = []

        i = 0
        # print(f'batches {len(train_loader)}')
        for batch in tqdm(train_loader, total=len(train_loader), desc='Epoch {}'.format(epoch)):
            i += 1
            optim.zero_grad()
            batch = prepare_batch(batch, text_only, patch, device, num_descriptions=num_captions, break_line=break_line)
            output = decoder(batch)
            loss = output.loss
            loss.backward()
            optim.step()
            loss = loss.detach().cpu().item()
            log_loss.append(loss)

            # logging and validation
            if (i + 1) % log_step == 0 or i == len(train_loader)-1:
                logging.debug('Logging step {}'.format(i + 1))
                # validation
                log_val_losses = []
                with torch.no_grad():
                    # noise may be used during training
                    decoder.add_noise = False
                    for val_batch in val_loader:
                        # validate using text embeddings in text only training
                        flag = True if dataset == 'petro-txt' else False
                        logging.debug(f'validation using text embedding? {flag}')
                        val_batch = prepare_batch(val_batch, flag, patch, device, num_descriptions=num_captions)

                        with torch.no_grad():
                            val_output = decoder(val_batch)
                            log_val_losses.append(val_output.loss.detach().cpu().item())

                # save step loss and clean list
                validation_losses.append(sum(log_val_losses) / len(log_val_losses))
                training_losses.append(sum(log_loss) / len(log_loss))
                log_loss = []

                # plot and save loss history
                plt.plot(range(len(training_losses)), training_losses, label='training')
                plt.plot(range(len(validation_losses)), validation_losses, label='validation')
                plt.legend()
                plt.xlabel('step')
                plt.ylabel('loss')
                plt.title(f'training loss')

                plt.savefig(f'{root}/loss_plot.png')

                plt.clf()
                log = {'training_loss': training_losses, 'validation_loss': validation_losses}
                with open(f'{root}/loss_log.pkl', 'wb') as f:
                    pickle.dump(log, f)

                decoder.train(True)
                decoder.add_noise = add_noise
                logging.debug(f'add noise to embeddings? {decoder.add_noise}')
                # model_size(decoder)
                # learnable_parameters(decoder)

        # epoch model
        model_dict = {'epoch': epoch + 1,
                      'model_state_dict': decoder.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': training_losses[-1]
                      }
        if save_history:
            torch.save(model_dict, f'{root}/checkpoint_{epoch+1}.pt')
        else:
            torch.save(model_dict, f'{root}/checkpoint.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--embeddings', type=str, required=True, help='embeddings filename')
    parser.add_argument('--rank', type=int, default=16, help='lora rank')
    parser.add_argument('--alpha', type=int, default=32, help='lora alpha parameter')
    parser.add_argument('--dropout', type=float, default=0.05, help='lora dropout parameter')
    parser.add_argument('--model_name', type=str, default="facebook/opt-350m", help='OPT model name')
    parser.add_argument('--prefix_len', type=int, default=10, help='model prefix length')
    parser.add_argument('--fp', choices=['fp16', 'fp32'], default='fp32', help='float precision')
    parser.add_argument('--text_only', action='store_true', default=False,
                        help='train using text embeddings as input instead of image embeddings')
    parser.add_argument('--full_finetune', action='store_true', help='fine tune entire model', default=False)
    parser.add_argument('--noise', action='store_true', help='add noise to embeddings', default=False)
    parser.add_argument('--variance', type=float, help='variance for noise injection', default=0.016)
    parser.add_argument('--history', action='store_true', help='save epoch history', default=False)
    parser.add_argument('--dataset', type=str, default='coco', help='dataset name',
                        choices=['coco', 'petro', 'petro-txt', 'cego', 'mimic'], )
    parser.add_argument('--save_path', required=True, help='root dir for saving results')
    parser.add_argument('--dimension', default=768, type=int, help='embedding dimension')
    parser.add_argument('--normalize', action='store_true', help='normalize embeddings', default=False)
    parser.add_argument('--log_step', type=int, default=None, help='log step')
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--patched', action='store_true', help='use patches', default=False)
    parser.add_argument('--before', action='store_true', help='prefix before begin of sentence token', default=False)
    parser.add_argument('--break_lines', action='store_true',
                        help='break string separated by new line, and user the first part only', default=False)
    parser.add_argument('--append_eos', action='store_true', default=False, help='append eos to the end of sentence')
    parser.add_argument('--pretrained_mapper', default=None, help='pretrained mapper checkpoint path')
    args = parser.parse_args()
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        logging.info(f'folders created: {args.save_path}')

    precision = torch.float16 if args.fp == 'fp16' else torch.float32
    logging.debug(f'precision: {precision}')
    cfg_path = os.path.join(args.model_name, 'adapter_config.json')

    # /home/users/adsdrosa/datasets / mimic / train_split_llava.json

    if os.path.exists(cfg_path):
        logging.debug('decoder was adapted before')
        with open(cfg_path, 'rb') as f:
            cfg = json.load(f)
            args.rank = cfg['r']
            args.alpha = cfg['lora_alpha']
            logging.debug(f'decoder rank: {args.rank} alpha: {args.alpha}')

    train(args.epochs, args.batch_size, args.lr, args.embeddings, args.rank, args.alpha, args.dropout,
          args.model_name, args.prefix_len, precision, args.text_only, args.full_finetune,
          args.noise, args.variance, args.history, args.dataset, args.save_path, args.dimension, args.log_step,
          args.normalize, args.patched, args.before, args.break_lines, args.append_eos, args.pretrained_mapper)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.save_path, 'checkpoint.pt')
    with open(f'{args.save_path}/experiment.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
        logging.info(f'experiment saved')
