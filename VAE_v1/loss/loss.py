import torch
import torch.nn.functional as F

def vae_loss(recon, target, mu, log_var):
    recon_loss = F.mse_loss(recon, target, reduction='mean')  # MAE au lieu de MSE
    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    beta = 1  # Facteur pour le KL, peut être ajusté
    
    return recon_loss + beta*kl_div
