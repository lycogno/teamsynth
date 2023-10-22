import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import DistilBertModel, BertTokenizer

import numpy as np

class BERTEncoder(nn.Module):
    def __init__(self, config, is_training = True):
        super(BERTEncoder, self).__init__()
        self.enc =  DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.att = nn.Linear(768, 1)
        self.fc = nn.Linear(768, config['embedding_size'])

    def forward(self, tokens):
        src = self.enc(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], return_dict=True)
        src = src.last_hidden_state

        wt = self.att(src)
        # wt = [batch size, src len, 1]
        wt = torch.softmax(wt, dim=1)
        src = torch.matmul(wt.permute(0, 2, 1), src)
        # src = [batch size, 1, 768]
        src = self.fc(src)
        # src = [batch size, 1, embed_size]
        src = torch.squeeze(src, dim=1)
        # src = [batch size, embed_size]
        return src


class APP(nn.Module):

    def __init__(self, config, is_training=True):
        super(APP, self).__init__()
        self._text_encoder = BERTEncoder(config, is_training)

        self.config = config

        self.fully_connected = nn.Sequential(
            nn.Linear(config['embedding_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
        )

        self._fc2 = nn.Linear(config['hidden_size'], config['pers_embedding_size'])

        self.mbti_classifier = nn.Linear(config['pers_embedding_size'], 16)

        self.OCEAN_layer = nn.Linear(config['pers_embedding_size'], config['ocean_size'])

        self.O_classifier = nn.Linear(config['ocean_size'], 2)
        self.C_classifier = nn.Linear(config['ocean_size'], 2)
        self.E_classifier = nn.Linear(config['ocean_size'], 2)
        self.A_classifier = nn.Linear(config['ocean_size'], 2)
        self.N_classifier = nn.Linear(config['ocean_size'], 2)

    def get_ocean_loss(self, predictions, labels):
        criterion = nn.CrossEntropyLoss()

        OCEAN_loss = 0
        for cat in ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']:
            OCEAN_loss += criterion(predictions[cat], labels[cat].long().to(device))

        return OCEAN_loss

    def get_mbti_loss(self, predictions, labels):
        criterion = nn.CrossEntropyLoss()

        mbti_loss = criterion(predictions['mbti'], torch.Tensor([labels]).long().to(device))
        return mbti_loss

    def forward(self, tokens):
        config = self.config
        # get text embeddings
        text_embeddings = self._text_encoder(tokens)

        # get personality embeddings
        personality_embeddings = self._fc2(F.relu(self.fully_connected(text_embeddings)))

        # get mbti
        mbti = self.mbti_classifier(F.relu(personality_embeddings))

        # get OCEAN
        OCEAN = F.relu(self.OCEAN_layer(F.relu(personality_embeddings)))
        O_pred = self.O_classifier(OCEAN)
        C_pred = self.C_classifier(OCEAN)
        E_pred = self.E_classifier(OCEAN)
        A_pred = self.A_classifier(OCEAN)
        N_pred = self.N_classifier(OCEAN)

        predictions = {
            'mbti': mbti,
            'cOPN': O_pred,
            'cCON': C_pred,
            'cEXT': E_pred,
            'cAGR': A_pred,
            'cNEU': N_pred,
            'personality_embeddings': personality_embeddings,
        }

        return predictions