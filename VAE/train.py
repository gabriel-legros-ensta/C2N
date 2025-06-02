import torch
from VAE.vae import VAE
from loss.loss import vae_loss 
from optimisation.optimisation import run_optuna
from data_processing.dataload import load_normalize_data
import json
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

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
dataset, scaler_x, scaler_y = load_normalize_data(batch_size=None, return_dataset=True)  # Adapter la fonction pour retourner le dataset complet

# Split du dataset en train et test (80%/20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Pour affichage des shapes
data_iter = iter(dataloader_train)
X_batch, y_batch = next(data_iter)
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
num_epochs = 300

# ============================================================
# Boucle d'entraînement
# ============================================================
train_losses = []

vae.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x_batch, y_batch in dataloader_train:
        xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)

        recon, mu, log_var = vae(xy_batch)
        loss_batch = vae_loss(recon, xy_batch, mu, log_var)

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        epoch_loss += loss_batch.item()

    avg_loss = epoch_loss / len(dataloader_train)
    train_losses.append(avg_loss)
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
# Évaluation du modèle sur le jeu de test
# ============================================================
vae.eval()
eval_loss = 0
n_batches = 0

with torch.no_grad():
    for x_batch, y_batch in dataloader_test:
        xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
        recon, mu, log_var = vae(xy_batch)
        loss = vae_loss(recon, xy_batch, mu, log_var)
        eval_loss += loss.item()
        n_batches += 1

eval_loss /= n_batches
print(f"Loss moyenne en test : {eval_loss:.4f}")

# ============================================================
# Sauvegarde du modèle entraîné
# ============================================================
model_path = "vae_trained.pth"
torch.save({
    'model_state_dict': vae.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'input_dim': input_dim,
    'latent_dim': latent_dim,
    'hidden_dim': hidden_dim,
    'dropout': dropout,
    'scaler_x': scaler_x,
    'scaler_y': scaler_y
}, model_path)
print(f"Modèle sauvegardé sous {model_path}")

# ============================================================
# Prédiction et reconstruction sur un batch de test
# ============================================================

vae.eval()
with torch.no_grad():
    # Prendre un batch de test
    x_batch, y_batch = next(iter(dataloader_test))
    xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)

    # Reconstruction
    recon, mu, log_var = vae(xy_batch)

    # Convertir en numpy
    recon_np = recon.cpu().numpy()
    original_np = xy_batch.cpu().numpy()

# Séparer X et y dans la reconstruction et les données originales

# Dans la partie prédiction, juste après recon_np et original_np
recon_y_norm = recon_np[:, dim_x:]
orig_y_norm  = original_np[:, dim_x:]

print("Y normalisé (orig) :", orig_y_norm[0])
print("Y normalisé (recon):", recon_y_norm[0])


recon_x = recon_np[:, :dim_x]
recon_y = recon_np[:, dim_x:]

orig_x = original_np[:, :dim_x]
orig_y = original_np[:, dim_x:]



# Inverse de la standardisation
recon_x_orig = scaler_x.inverse_transform(recon_x)
recon_y_orig = scaler_y.inverse_transform(recon_y)

orig_x_orig = scaler_x.inverse_transform(orig_x)
orig_y_orig = scaler_y.inverse_transform(orig_y)

# ============================================================
# Affichage d'exemples (ici on affiche la première donnée du batch)
# ============================================================
index = 0

print("=== Données originales ===")
print("X:", orig_x_orig[index])
print("Y:", orig_y_orig[index])

print("\n=== Reconstruction ===")
print("X reconstruit:", recon_x_orig[index])
print("Y reconstruit:", recon_y_orig[index])

# Optionnel: tu peux aussi faire des plots si tes données sont visuelles ou graphiques
# Par exemple, si X est une image ou un vecteur 2D:
# plt.figure()
# plt.plot(orig_x_orig[index], label='Original X')
# plt.plot(recon_x_orig[index], label='Recon X')
# plt.legend()
# plt.show()
