import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc31 = nn.Linear(512, latent_dim)  # μ
        self.fc32 = nn.Linear(512, latent_dim)  # logσ²

    def forward(self, xy):
        h = F.relu(self.fc1(xy))
        h = F.relu(self.fc2(h))
        mu = self.fc31(h)
        log_var = self.fc32(h)
        return mu, log_var
    