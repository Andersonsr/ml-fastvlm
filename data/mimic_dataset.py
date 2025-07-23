import json
import os
import logging
import glob
import torch
import numpy as np
import pickle
import gc
from PIL import Image
from torchvision.transforms import ToTensor

to_tensor = ToTensor()


class MimicDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, json_file, zeroed=False):
        assert os.path.exists(json_file), '{} does not exist'.format(json_file)
        assert os.path.isdir(root_dir), '{} is not a dir'.format(root_dir)

        self.root = root_dir
        self.data = json.load(open(json_file, 'r'))
        self.zeroed = zeroed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.root, self.data[i]['image_name']))
        image = to_tensor(image)
        return {'id': self.data[i]['id'],
                'findings': self.data[i]['findings'],
                'labels': self.data[i]['labels'],
                'image': image,
                'patient': self.data[i]['patient'],
                'study': self.data[i]['study']}

    def collate_fn(self, batch):
        data = {}

        for sample in batch:
            for key, item in sample.items():
                if key not in data.keys():
                    data[key] = []

                data[key].append(item)

        data['image'] = torch.stack(data['image'], dim=0)
        # reorganize labels
        reorganized_labels = {}
        for key in data['labels'][0].keys():
            reorganized_labels[key] = []

        for labels in data['labels']:
            for key in reorganized_labels.keys():
                # print('old', labels[key])
                # print('new', 0 if labels[key] == 3 and self.zeroed else labels[key])
                reorganized_labels[key].append(0 if self.zeroed and labels[key] == 3 else labels[key])

        data['labels'] = reorganized_labels
        return data

    def get_loader(self, batch_size):
        indices = np.arange(len(self))
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=False,
                                           collate_fn=self.collate_fn)


class MimicChunkDataset(torch.utils.data.Dataset):
    def __init__(self, dirname, chunks=None, unchanged_labels=False):
        assert os.path.exists(dirname), '{} does not exist'.format(dirname)
        assert os.path.exists(os.path.join(dirname, 'info.json'))
        if os.path.isdir(dirname):
            logging.debug('searching for files in {}'.format(dirname))
            self.chunks = glob.glob(os.path.join(dirname, '*.pkl'))
            assert len(self.chunks) > 0, 'No .pkl files found in {}'.format(dirname)
            logging.debug('found {} chunks'.format(len(self.chunks)))

        else:
            logging.debug('single chunk {}'.format(dirname))
            self.chunks = [dirname]

        self.chunks.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        if chunks is not None:
            assert chunks < len(self.chunks), '{} exceeds number of chunks'.format(chunks)
            self.chunks = self.chunks[:chunks]

        self.unchanged_labels = unchanged_labels
        self.len = 0
        with open(os.path.join(dirname, 'info.json'), 'r') as f:
            info = json.load(f)
            self.len = info['total_images']

        logging.debug('total number of images: {}'.format(self.len))

        self.data = {}
        self.current_chunk = 0
        self.offset = 0
        self.limit = 0

    def free_data(self):
        # free memory to load next chunk
        self.data = None
        gc.collect()

    def load_chunk(self, index):
        assert 0 <= index <= len(self.chunks), 'index out of range'
        logging.debug('loading chunk {}'.format(index))
        with open(self.chunks[index], 'rb') as f:
            if index == 0:
                # reset chunks
                self.current_chunk = 0
                self.offset = 0
                self.free_data()
                self.data = pickle.load(f)
                self.limit = len(self.data['image_ids'])

            else:
                assert index == self.current_chunk + 1, 'chunks must be loaded in order'
                # loading next chunk
                self.current_chunk = index
                self.offset += len(self.data['image_ids'])
                self.free_data()
                self.data = pickle.load(f)
                self.limit += len(self.data['image_ids'])

        logging.debug(f'limit {self.limit}, offset {self.offset}, current chunk {self.current_chunk}')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index == 0:
            self.free_data()
            self.load_chunk(0)

        elif index >= self.limit:
            if self.current_chunk == len(self.chunks) - 1:
                # last chunk, reset iteration
                self.load_chunk(0)

            else:
                # load next chunk
                self.load_chunk(self.current_chunk + 1)

        payload = {}
        for key in self.data.keys():
            payload[key] = self.data[key][index - self.offset]

        if 'image_tensors' in payload.keys():
            payload['image_tensors'] = to_tensor(payload['image_tensors'])

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
        if 'image_tensors' in data.keys():
            data['image_tensors'] = torch.stack(data['image_tensors'])
        return data


if __name__ == '__main__':
    dataset = MimicDataset('E:\\datasets\\mimic\\preprocess\\resize_1024', 'E:\\datasets\\mimic\\preprocess\\train_split.json', zeroed=True)
    loader = dataset.get_loader(4)
    from tqdm import tqdm
    for batch in tqdm(loader):
        # print(batch['labels'])
        pass
