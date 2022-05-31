import pandas as pd
from torch.utils.data import Dataset
import json

class StreamDataset(Dataset):
    def __init__(self, path):
        self.contexts = []
        self.statements = []
        self.labels = []
        df = pd.read_csv(path)

        for i in range(len(df)):
            self.contexts.append(df['context'][i])
            self.statements.append(df['statement'][i])
            self.labels.append(df['label'][i])

        self.n_examples = len(self.labels)

        return

    def __len__(self):
        return self.n_examples

    def __getitem__(self, item):
        return {'context':self.contexts[item], 'statement':self.statements[item], 'label':self.labels[item]}


class MBPADataset(Dataset):
    def __init__(self, path):
        self.contexts = []
        self.statements = []
        self.labels = []
        df = pd.read_csv(path)

        with open(path) as f:
            json_data = json.load(f)
            items = [_ for _ in json_data['data']]

        for item in items:
            item = item['paragraphs'][0]
            self.contexts.append(item['context'])
            self.statements.append(item['qas'][0]['question'])
            self.labels.append(item['qas'][0]['answers'][0]['text'])

        self.n_examples = len(self.labels)

        return

    def __len__(self):
        return self.n_examples

    def __getitem__(self, item):
        return {'context':self.contexts[item], 'statement':self.statements[item], 'label':self.labels[item]}
    
