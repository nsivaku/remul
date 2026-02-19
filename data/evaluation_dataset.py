from datasets import load_dataset
import json


class Dataset:

    def __init__(self, name):
        self.name = name
        if name == "bbeh":
            with open('datasets/bbeh/bbeh.json', 'r') as f:
                self.data = json.load(f)
        elif name == "zlb":
            with open('datasets/zebralogicbench/zebralogicbench.json', 'r') as f:
                self.data = json.load(f)
        elif name == "musr":
            with open('datasets/musr/musr.json', 'r') as f:
                self.data = json.load(f)
        elif name == "folio":
            with open('datasets/folio/folio.json', 'r') as f:
                self.data = json.load(f)
    
    # define index of the dataset
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
