import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        out = self.fc3(h)
        return out
