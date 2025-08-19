import glob
import os.path
import pickle
import math
from tqdm import tqdm
from torchvision import transforms
import logging
import time
import gc
import json
import numpy as np
import wget
from PIL import Image
from model.classifiers import mimic_classifier_list
# pil to tensor
to_tensor = transforms.ToTensor()
import torch


def fix_mimic_chunk_labels(filename):
    '''
    edit chunk file with reorganized labels
    :param filename: chunk pkl file to edit
    :return: None
    '''
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for i, e in enumerate(data['labels']):
            # old labels: positive: 1, negative: 0, uncertain: -1, ignore: nan
            # new labels: positive: 1, negative: 0, uncertain: 2, ignore: 3
            new_labels = {}
            for label in mimic_classifier_list:
                if label not in e.keys():
                    new_labels[label] = 3
                    print('image {} missing label: {}'.format(i, label))
                else:
                    if math.isnan(e[label]):
                        new_labels[label] = 3
                    else:
                        new_labels[label] = 2 if e[label] < 0 else e[label]
            data['labels'][i] = new_labels

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def count_length(dirname):
    total_len = 0
    cache = {}
    assert os.path.exists(dirname), '{} does not exist'.format(dirname)
    chunks = glob.glob(os.path.join(dirname, "chunk*.pkl"))
    assert len(chunks) > 0, "No chunks found at {}".format(dirname)
    for chunk in tqdm(chunks):
        logging.debug('loading chunk {} ...'.format(chunk))
        starting_time = time.time()
        with open(chunk, 'rb') as f:
            data = None
            gc.collect()
            data = pickle.load(f)
            ending_time = time.time()
            logging.debug('load time: {}'.format(ending_time - starting_time))
            length = len(data['image_name'])

        logging.info('{} length is {}'.format(chunk, length))
        cache[os.path.basename(chunk)] = length
        total_len += length

    with open(os.path.join(dirname, 'data_length.json'), 'w') as f:
        json.dump(cache, f)


def mimic_stats():
    studs = glob.glob('E:\\datasets\\mimic\\mimic-cxr-jpg\\2.1.0\\files\\*\\*\\*')
    images_per_study = []
    dim = [1]
    for study in tqdm(studs):
        images = glob.glob(os.path.join(study, '*.jpg'))
        images_per_study.append(len(images))
        for image in images:
            image = Image.open(image)
            w, h = image.size
            dim.append(min(w, h))

    dict = {}
    print('images per study')
    uniques, counts = np.unique(images_per_study, return_counts=True)
    for unique, count in zip(uniques, counts):
        print('{}: {}'.format(unique, count))
        dict[str(unique)] = int(count)

    print('total: {}'.format(np.sum(counts)))
    print()
    print('image dimension')
    print('average image dimension: {}'.format(np.mean(dim)))
    print('std image dimension: {}'.format(np.std(dim)))
    print('median image dimension: {}'.format(np.median(dim)))
    print('max image dimension: {}'.format(np.max(dim)))
    print('min image dimension: {}'.format(np.min(dim)))

    with open('../mimic_stats.json', 'w') as f:
        json.dump({'images_per_study': dict, 'dimensions': dim}, f, indent=2)


def check_mimic_download():

    root = 'E:\\datasets\\mimic\\mimic-cxr-jpg\\2.1.0\\'
    downloaded_files = glob.glob(os.path.join(root, 'files\\*\\*\\*\\*.jpg'), recursive=True)
    print(downloaded_files[0])
    to_download = []
    files = open('E:\\datasets\\mimic\\IMAGE_FILENAMES', 'r')
    for filename in tqdm(files):
        if not os.path.exists(os.path.join(root, filename.replace('\n', ''))):
            to_download.append(filename)

    print('to download', len(to_download))
    print('downloaded files: ', len(downloaded_files))

    open('E:\\datasets\\mimic\\to_download', 'w').write(''.join(to_download))


def check_mimic_max_resolution():
    root = 'E:\\datasets\\mimic\\mimic-cxr-jpg\\2.1.0\\'
    downloaded_files = glob.glob(os.path.join(root, 'files\\*\\*\\*\\*.jpg'), recursive=True)
    max_resolution = 0
    dimensions = (0, 0)
    filename = ''
    for i, entry in tqdm(enumerate(downloaded_files), total=len(downloaded_files)):
        image = Image.open(entry)

        width, height = image.size
        if width * height > max_resolution:
            max_resolution = width * height
            filename = entry
            dimensions = (width, height)

            print(f'{i} new max resolution: ', dimensions)

    print(filename)
    print(dimensions)
    # (4280, 3520)


def mimic_classification_plot():
    models = ['class-4-mixer-lora', 'class-4-full', 'class-4-full-noavg']
    for model in models:
        eval = json.load(open(f'checkpoints/{model}/classification_eval.json', 'r'))
        for e in eval:
            print(e)
        break


def plot_f1():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    results = {'experiment': [], 'condition': [], 'f1-score': [], 'recall': [], 'precision': []}

    for name in ['class-3-mixer-lora train', 'class-3-mixer-lora test']:
        split = name.split()
        if len(split) > 1:
            data = json.load(open(f'checkpoints/{split[0]}/classification_eval_{split[1]}.json'))

        else:
            data = json.load(open(f'checkpoints/{name}/classification_eval.json'))

        for e in data:
            results['experiment'].append(name)
            results['condition'].append(e['condition'])
            results['f1-score'].append(e['report']['macro avg']['f1-score'])
            results['recall'].append(e['report']['macro avg']['recall'])
            results['precision'].append(e['report']['macro avg']['precision'])

    df = pd.DataFrame(results)
    ax = sns.barplot(df, y='condition', x='f1-score', hue='experiment')
    ax.legend(bbox_to_anchor=(0.25, 1.05), loc='lower left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_f1()


