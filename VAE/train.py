import torch
# from VAE.vae import VAE
from loss import loss
from optimisation.optimisation import run_optuna
from data_processing.dataload import load_normalize_data
import json

# ============================================================
# Chargement des meilleurs paramètres
# ============================================================
with open("best_params.json", "r") as f:
    best_params = json.load(f)

latent_dim = best_params["latent_dim"]
hidden_dim = best_params["hidden_dim"]
lr = best_params["lr"]
print("Best parameters found:")
print("Latent dimension:", latent_dim)
print("Hidden dimension:", hidden_dim)
print("Learning rate:", lr)

# ============================================================
# Chargement et normalisation des données
# ============================================================
dataloader, scaler_x, scaler_y = load_normalize_data(batch_size=64)

data_iter = iter(dataloader)
X_batch, y_batch = next(data_iter)  # X_batch shape = (batch_size, dim_x), y_batch shape = (batch_size, dim_y)
print("X_batch shape:", X_batch.shape)
print("y_batch shape:", y_batch.shape)

dim_x = X_batch.shape[1]
dim_y = y_batch.shape[1]
print("dim_x:", dim_x)
print("dim_y:", dim_y)

input_dim = dim_x + dim_y

# ============================================================
# Configuration du device
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ============================================================
# Initialisation du modèle et de l'optimiseur
# ============================================================
vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
num_epochs = 10

# ============================================================
# Boucle d'entraînement simplifiée
# ============================================================
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)

        recon, mu, log_var = vae(xy_batch)
        loss = vae_loss(recon, xy_batch, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
