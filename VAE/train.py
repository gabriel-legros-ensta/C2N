import torch
from VAE.vae import VAE
from loss.loss import vae_loss 
from optimisation.optimisation import run_optuna
from data_processing.dataload import load_normalize_data
import json
import matplotlib.pyplot as plt  # Ajout pour l'affichage de la courbe

# ============================================================
# Chargement des meilleurs paramètres
# ============================================================
with open("best_params.json", "r") as f:
    best_params = json.load(f)

latent_dim = best_params["latent_dim"]
hidden_dim = best_params["hidden_dim"]
lr = best_params["lr"]
dropout = best_params["dropout"]
print("Best parameters found:")
print("Latent dimension:", latent_dim)
print("Hidden dimension:", hidden_dim)
print("Learning rate:", lr)
print("Dropout rate:", dropout)
# ============================================================
# Chargement et normalisation des données
# ============================================================
dataloader, scaler_x, scaler_y = load_normalize_data(batch_size=32)

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
vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
num_epochs = 200

# ============================================================
# Boucle d'entraînement simplifiée
# ============================================================
train_losses = []  # Liste pour stocker la loss de chaque epoch

vae.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x_batch, y_batch in dataloader:
        xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)

        recon, mu, log_var = vae(xy_batch)
        loss_batch = vae_loss(recon, xy_batch, mu, log_var)

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        epoch_loss += loss_batch.item()

    avg_loss = epoch_loss / len(dataloader)
    train_losses.append(avg_loss)  # Stocke la loss moyenne de l'epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ============================================================
# Affichage de la courbe de loss d'entraînement
# ============================================================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Courbe de la loss d'entraînement")
plt.legend()
plt.show()

# ============================================================
# Évaluation du modèle après entraînement
# ============================================================
vae.eval()
eval_loss = 0
n_batches = 0

with torch.no_grad():
    for x_batch, y_batch in dataloader:  # ou un dataloader de validation
        xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
        recon, mu, log_var = vae(xy_batch)
        loss = vae_loss(recon, xy_batch, mu, log_var)
        eval_loss += loss.item()
        n_batches += 1

eval_loss /= n_batches
print(f"Loss moyenne en évaluation : {eval_loss:.4f}")


