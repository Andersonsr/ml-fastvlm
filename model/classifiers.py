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
        self.classifiers_list = classifiers_list
        for classifier in classifiers_list:
            self.add_module(classifier, LinearClassifier(input_size, output_classes))

    def forward(self, x):
        y = {}
        for name, module in self.named_children():
            y[name] = module(x)
        return y





