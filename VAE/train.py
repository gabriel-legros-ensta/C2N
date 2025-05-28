import torch
#from VAE.vae import VAE
from loss import loss
from data_processing.dataload import load_normalize_data


dataloader, scaler_x, scaler_y = load_normalize_data(batch_size=64)






# Dimensions
dim_x = 4
dim_y = 5000
input_dim = dim_x + dim_y
latent_dim = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = VAE(input_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
num_epochs = 10

# Extrait simplifié de boucle d'entraînement
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)

        recon, mu, log_var = vae(xy_batch)
        loss = vae_loss(recon, xy_batch, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
