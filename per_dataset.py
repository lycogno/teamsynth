import torch
from torch.utils.data import Dataset
import numpy as np
import json

import pandas as pd

from transformers import BertTokenizer

class Essays_dataset(Dataset):
 
    def __init__(self, file_name):
        df = pd.read_csv(file_name ,encoding='latin1')

        self.tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
        
        cat_columns = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
        df[cat_columns] = df[cat_columns].replace({'y': 1, 'n': 0})

        self.X = df.drop(cat_columns, axis=1)
        self.y = df[cat_columns]
 
    def __len__(self):
        return len(self.X)
   
    def __getitem__(self,idx):
        text = self.X.iloc[idx]['TEXT']
        tokens = self.tokeniser(text, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
        # tokens = {'input_ids': tokens['input_ids'], 'token_type_ids': tokens['token_type_ids']}
        tokens['input_ids'] = tokens['input_ids'].squeeze(0)
        tokens['token_type_ids'] = tokens['token_type_ids'].squeeze(0)
        tokens['attention_mask'] = tokens['attention_mask'].squeeze(0)
        labels = self.y.iloc[idx].to_dict()

        return tokens, labels