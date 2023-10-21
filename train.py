# write a training loop to train the APP model
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from APP_model import *

import json

ESSAYS_PATH = 'datasets/essays.csv'
MBTI_PATH = 'datasets/mbti_1.csv'

def load_data(essays_path, mbti_path):
    essays_df = pd.read_csv(ESSAYS_PATH, encoding='latin1')
    cat_columns = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
    essays_df[cat_columns] = essays_df[cat_columns].replace({'y': 1, 'n': 0})

    mbti_df = pd.read_csv(MBTI_PATH)
    types = mbti_df['type'].unique()
    types = dict(zip(types, range(len(types))))
    mbti_df['type'] = mbti_df['type'].replace(types)

    return essays_df, mbti_df

essays_df, mbti_df = load_data(ESSAYS_PATH, MBTI_PATH)

epochs = 1

config = json.load(open('config.json'))

model = APP(config)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

cat_columns = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']

running_loss_essay = 0
running_loss_mbti = 0

# profile the code using cProfile


def training_loop(model, data):
    essays_df, mbti_df = data
    for i in range(epochs):
        for index, row in essays_df.iterrows():
            text = row['TEXT']
            labels = row[cat_columns].to_dict()
            
            optimizer.zero_grad()

            preds = model(text)
            loss = model.get_ocean_loss(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss_essay += loss.item()

            if index % 1 == 0:    # print every 2000 mini-batches
                print(f'essay index: {index + 1}, epoch: {i + 1} loss: {running_loss_essay / 100:.3f}')
                running_loss_essay = 0.0

        for index, row in mbti_df.iterrows():
            text = row['posts']
            labels = row['type']

            optimizer.zero_grad()

            preds = model(text)
            loss = model.get_mbti_loss(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss_mbti += loss.item()

            if index % 100 == 1:    # print every 2000 mini-batches
                print(f'mbti index: {index + 1}, epoch: {i + 1} loss: {running_loss_mbti / 100:.3f}')
                running_loss_mbti = 0.0
        
