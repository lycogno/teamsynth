import torch.nn as nn
import torch.nn.functional as F
import torch

class SentimentClassifier(nn.Module):
    def __init__(self, config, is_training=True):
        super().__init__()
        self.input_size = config['pers_embedding_size']
        self.fc1 = nn.Linear(config['pers_embedding_size'], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

