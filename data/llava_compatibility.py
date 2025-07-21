import json
from tqdm import tqdm


if __name__ == '__main__':
    for name in ['train_split_filter', 'dev_split_filter', 'dev_split', 'test_split', 'train_split']:
        filename = f"E:\\datasets\\mimic\\preprocess\\{name}.json"
        data = json.load(open(filename, 'r'))
        new_data = []
        for sample in tqdm(data):
            new_sample = {'image': sample['image_name'], 'id': sample['id'], 'conversations': []}
            # new_sample['conversations'].append({'from': 'system', 'value': 'You are a medical AI assistant specialized in interpreting chest radiographs\n.'})
            new_sample['conversations'].append({'from': 'human', 'value': 'describe the findings in the image\n<image>'})
            new_sample['conversations'].append({'from': 'gpt', 'value': sample['findings']})
            new_data.append(new_sample)

        json.dump(new_data, open(f'E:\\datasets\\mimic\\preprocess\\{name}_llava.json', 'w'), indent=2)

