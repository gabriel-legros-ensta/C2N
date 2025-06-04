import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_list, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim_list=hidden_dim_list, dropout=dropout)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim_list=hidden_dim_list[::-1], dropout=dropout)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, xy):
        mu, log_var = self.encoder(xy)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var
