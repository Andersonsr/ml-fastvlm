import json
from tqdm import tqdm


if __name__ == '__main__':
    filename = "E:\\datasets\\mimic\\preprocess\\train_split.json"
    data = json.load(open(filename, 'r'))
    new_data = []
    for sample in tqdm(data):
        new_sample = {'image': sample['image_name'], 'id': sample['id'], 'conversations': []}
        # new_sample['conversations'].append({'from': 'system', 'value': '\nYou are a medical AI assistant specialized in interpreting chest radiographs.'})
        new_sample['conversations'].append({'from': 'human', 'value': 'describe the findings in <image>'})
        new_sample['conversations'].append({'from': 'gpt', 'value': sample['findings']})
        new_data.append(new_sample)

    json.dump(new_data, open('E:\\datasets\\mimic\\preprocess\\micro_split_llava.json', 'w'), indent=2)

