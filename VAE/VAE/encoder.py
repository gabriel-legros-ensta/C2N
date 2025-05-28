import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)  # μ
        self.fc32 = nn.Linear(hidden_dim, latent_dim)  # logσ²
        self.dropout = nn.Dropout(dropout)

    def forward(self, xy):
        h = F.relu(self.fc1(xy))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        mu = self.fc31(h)
        log_var = self.fc32(h)
        return mu, log_var
