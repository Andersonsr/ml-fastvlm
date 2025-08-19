import argparse
import gc
import glob
import json
import logging
from tqdm import tqdm
import os.path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class PetroDataset(Dataset):
    def __init__(self, path, split=None, ratio=0.9):
        '''
        Load dataset from file and split into training or validation sets, ratio should be between 0 and 1
        :param path: path to dataset
        :param split: None to use the whole dataset, train or val
        :param ratio: train ratio
        '''
        assert os.path.exists(path), '{} does not exist'.format(path)
        data = pickle.load(open(path, 'rb'))
        lim = int(ratio * len(data['image_id']))
        # print(data)
        self.patch_embeddings = []
        if split is None:
            self.text_embeddings = data['text_embeddings']
            self.image_embeddings = data['image_embeddings']
            self.captions = data['captions']
            self.image_id = data['image_id']
            if 'patch_embeddings' in data:
                self.patch_embeddings = data['patch_embeddings']

        elif split == 'train':
            self.text_embeddings = data['text_embeddings'][:lim]
            self.image_embeddings = data['image_embeddings'][:lim]
            self.captions = data['captions'][:lim]
            self.image_id = data['image_id'][:lim]
            if 'patch_embeddings' in data:
                self.patch_embeddings = data['patch_embeddings'][:lim]

        elif split == 'val':
            self.text_embeddings = data['text_embeddings'][lim:]
            self.image_embeddings = data['image_embeddings'][lim:]
            self.captions = data['captions'][lim:]
            self.image_id = data['image_id'][lim:]
            if 'patch_embeddings' in data:
                self.patch_embeddings = data['patch_embeddings'][lim:]

        else:
            raise ValueError('{} is not a valid split, choices = [train, val]'.format(split))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        payload = {'image_id': self.image_id[index],
                   'image_embeddings': self.image_embeddings[index],
                   'text_embeddings': self.text_embeddings[index],
                   'captions': self.captions[index]}
        if len(self.patch_embeddings) > 0:
            payload['patch_embeddings'] = self.patch_embeddings[index]

        return payload

    def get_loader(self, batch_size):
        '''
        get torch dataloader
        :param batch_size: batch size for the dataloader
        :return: dataloader, indices
        '''
        indices = np.arange(len(self.image_embeddings))
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)


class MIMICLoader(Dataset):
    def __init__(self, file,):
        self.data = pickle.load(open(file, 'rb'))
        self.len = len(self.data['image_embeddings'])
        print(self.data.keys(), self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        payload = {}
        for key in self.data.keys():
            if key != "labels":
                payload[key] = self.data[key][index]

        return payload

    def get_loader(self, batch_size):
        indices = np.arange(self.len)
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=False,
                                           collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        data = {}
        for e in batch:
            for key in e.keys():
                if key not in data.keys():
                    data[key] = []

                data[key].append(e[key])
        new_labels = {}
        if 'image_embeddings' in data.keys():
            # loaded a pkl with embeddings
            data['image_embeddings'] = torch.stack(data['image_embeddings'])
            data['text_embeddings'] = torch.stack(data['text_embeddings'])

        # if 'image_tensor' in data.keys():
        #     h, w = data['image_tensor'][0].size
        #     b = len(data['image_tensor'])
        #     images = np.asarray(data['image_tensor']).reshape((b, 3, w, h))
        #     logging.debug('batch image shape {}'. format(images.shape))
        #     data['image_tensor'] = torch.tensor(images)

        return data


class COCODataset(Dataset):
    def __init__(self, path, n_captions=5):
        assert os.path.exists(path), '{} does not exist'.format(path)
        self.text_embeddings = []
        self.captions = []
        self.patch_embeddings = []
        with open(path, 'rb') as f:
            data = pickle.load(f)
            logging.debug('data keys: {}'.format(data.keys()))
            logging.debug('len captions : {}'.format(len(data['captions'])))
            logging.debug('len images : {}'.format(len(data['image_embeddings'])))

            for i in range(len(data['text_embeddings'])):
                self.text_embeddings.append(data['text_embeddings'][i][:n_captions])
                self.captions.append(data['captions'][i][:n_captions])

            self.image_embeddings = data['image_embeddings']
            self.image_id = data['image_id']
            self.image_name = data['image_name']
            if 'patch_embeddings' in data.keys():
                self.patch_embeddings = data['patch_embeddings']

        # print(len(self.text_embeddings), len(self.captions))
        # print(self.image_id)

    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, index):
        # print('embedding shape 0', self.image_embeddings[index].shape)
        payload = {'image_id': self.image_id[index],
                   'image_name': self.image_name[index],
                   'image_embeddings': self.image_embeddings[index],
                   'text_embeddings': self.text_embeddings[index],
                   'captions': self.captions[index]}
        if len(self.patch_embeddings) > 0:
            payload['patch_embeddings'] = self.patch_embeddings[index]

        return payload

    def collate_fn(self, batch):
        ids = []
        names = []
        image_embeddings = []
        text_embeddings = []
        captions = []
        patch_embeddings = []
        for e in batch:
            ids.append(e['image_id'])
            names.append(e['image_name'])
            image_embeddings.append(e['image_embeddings'])
            text_embeddings.append(e['text_embeddings'])
            captions.append(e['captions'])
            if 'patch_embeddings' in e.keys():
                # print(e['patch_embeddings'].shape)
                patch_embeddings.append(e['patch_embeddings'].squeeze(1))

        payload = {'image_id': ids,
                   'image_name': names,
                   'image_embeddings': torch.stack(image_embeddings),
                   'text_embeddings': torch.stack(text_embeddings),
                   'captions': captions}

        if len(patch_embeddings) > 0:
            payload['patch_embeddings'] = torch.stack(patch_embeddings)

        return payload

    def get_loader(self, shuffle=False, batch_size=400):
        indices = np.arange(len(self.image_embeddings))
        if shuffle:
            np.random.shuffle(indices)
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False,
                                           collate_fn=self.collate_fn)

    def get_image_means(self):
        embeds = torch.stack(self.image_embeddings)
        means = torch.zeros(1, embeds.shape[-1])
        for e in embeds:
            means += e / e.norm(dim=-1, keepdim=True)
        means = means / embeds.shape[0]
        return means

    def get_text_means(self):
        embeds = []
        for e in self.text_embeddings:
            embeds += e
        embeds = torch.stack(embeds)
        means = torch.zeros(1, embeds.shape[-1])
        for e in embeds:
            means += e / e.norm(dim=-1, keepdim=True)
        means = means / embeds.shape[0]
        return means


if __name__ == '__main__':
    # debugging only
    parser = argparse.ArgumentParser('check output format')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name',
                        choices=['coco', 'mimic', 'petro', 'cego'])
    parser.add_argument('--path', type=str, required=True, help='path to dataset')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG)

    if args.dataset == 'coco':
        dataset = COCODataset(args.path)
    elif args.dataset == 'petro':
        dataset = PetroDataset(args.path, split='train')
    elif args.dataset == 'mimic':
        dataset = MIMICLoader(args.path)
    else:
        raise ValueError('dataset not supported, choices=[petro, coco]')

    loader = dataset.get_loader(batch_size=4)
    print(f'batches: {len(loader)}')
    for epoch in range(3):
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            if i == 0 and epoch == 0:
                if 'image_embeddings' in batch.keys():
                    print('image embeddings', batch['image_embeddings'].shape)
                    print('text embeddings', batch['text_embeddings'].shape)
                print(batch)
            break

