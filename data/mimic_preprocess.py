import os.path
import math
from PIL import Image, ImageFile
from tqdm import tqdm
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess(img, crop, size):
    if not crop:
        width, height = img.size
        # resize bigger images
        if max(width, height) > size:
            if height > width:
                new_height = size
                new_width = width * new_height / height
            else:
                new_width = size
                new_height = height * new_width / width
            img = img.resize((int(new_width), int(new_height)))
        # pad image so all will keep the same size
        new_image = Image.new('RGB', (size, size), (0, 0, 0))
        top_left = (size - img.size[0]) // 2, (size - img.size[1]) // 2
        Image.Image.paste(new_image, img, top_left)
        return new_image


if __name__ == '__main__':
    filename = 'E:\\datasets\\mimic\\chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json'
    output_dir = 'E:\\datasets\\mimic\\preprocess\\resize_1024_test'
    root = 'E:\\datasets\\mimic\\mimic-cxr-jpg\\2.1.0\\files\\'
    size = 1024
    crop = False
    data = []
    os.makedirs(output_dir, exist_ok=True)
    file = json.load(open(filename, 'r'))
    for i, sample in tqdm(enumerate(file), total=len(file)):
        new_labels = {}
        if sample['generate_method'] == 'gpt4':
            path = os.path.join(root, sample['image'].replace('mimic/', ''))
            if os.path.exists(path):
                split = sample['image'].split('/')
                image_name = split[-1]
                study = split[-2]
                patient = split[-3]

                if not os.path.exists(os.path.join(output_dir, image_name)):
                    im = Image.open(path).convert('RGB')
                    im = preprocess(im, crop, size)
                    im.save(os.path.join(output_dir, image_name))

                # new annotation
                for key, item in sample['chexpert_labels'].items():
                    if math.isnan(item):
                        new_labels[key] = 3
                    else:
                        new_labels[key] = 2 if item < 0 else item

                data.append({'id': sample['id'],
                             'findings': sample['conversations'][1]['value'].replace('\n', ''),
                             'labels': new_labels,
                             'image_name': image_name,
                             'patient': patient,
                             'study': study})
            else:
                print('missing file {}'.format(path))
    with open(os.path.join(os.path.dirname(output_dir), 'test_split.json'), 'w') as outfile:
        json.dump(data, outfile, indent=2)


