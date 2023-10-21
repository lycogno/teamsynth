import torch
from torch.utils.data import Dataset

import nltk
import numpy as np
import json

import pandas as pd

from transformers import BertTokenizer

class Essays_dataset(Dataset):
 
    def __init__(self, file_name):
        df = pd.read_csv(file_name ,encoding='latin1')
        
        cat_columns = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
        df[cat_columns] = df[cat_columns].replace({'y': 1, 'n': 0})

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def encode_and_pad_sentences_bert(self, sentences, max_sents_per_image, max_sent_len):
        """Encodes and pads sentences.

        Args:
            sentences: a list of python string.
            max_sents_per_image: maximum number of sentences.
            max_sent_len: maximum length of sentence.

        Returns:
            num_sents: a integer denoting the number of sentences.
            sent_mat: a [max_sents_per_image, max_sent_len] numpy array pad with zero.
            sent_len: a [max_sents_per_image] numpy array indicating the length of each
            sentence in the matrix.
            """

        sentences = [list(torch.squeeze(self.bert(s, return_tensors="pt")['input_ids'], dim=0).numpy()) for s in sentences]
        sent_mat = np.zeros((max_sents_per_image, max_sent_len), np.int32)
        sent_len = np.zeros((max_sents_per_image,), np.int32)

        for index, sent in enumerate(sentences[:max_sents_per_image]):
            sent_len[index] = min(max_sent_len, len(sent))
            sent_mat[index][:sent_len[index]] = sent[:sent_len[index]]

        return len(sentences), sent_mat, sent_len
 
    def __len__(self):
        return len(self.y_train)
   
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]
