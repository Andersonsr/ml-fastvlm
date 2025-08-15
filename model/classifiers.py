import torch
from torch import nn
# removed 'No Finding'
mimic_classifier_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                         'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
                         'Pneumonia', 'Pneumothorax', 'Support Devices']


class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_classes):
        super(LinearClassifier, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_size, output_classes))

    def forward(self, x):
        return self.mlp(x)


class MultiClassifier(nn.Module):
    def __init__(self, classifiers_list, input_size, output_classes):
        super(MultiClassifier, self).__init__()
        for name in classifiers_list:
            self.add_module(name, LinearClassifier(input_size, output_classes))

    def forward(self, x):
        y = {}
        if len(x.shape) == 2:
            for name, module in self.named_children():
                # print(type(module))
                y[name] = module(x)
            return y

        if len(x.shape) == 3:
            # mapper output
            for i, child in enumerate(self.named_children()):
                name, module = child
                # print(x[:, i, :].shape)
                y[name] = module(x[:, i, :])
            return y


if __name__ == '__main__':
    model = MultiClassifier(mimic_classifier_list, 896, 4)
    input = torch.rand((16, 13, 896))
    output = model(input)
    print(output)

