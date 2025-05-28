import optuna
import torch
from model import VAE  # ton module contenant Encoder, Decoder, VAE
from train_utils import train_vae  # fonction pour entraîner un VAE et retourner la loss
from .data_processing.dataload import load_normalize_data

def objective(trial):
    # Hyperparamètres à optimiser
    latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

    # Charger les données
    dataloader, _, _ = load_normalize_data(batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Créer le modèle avec les paramètres proposés
    input_dim = next(iter(dataloader))[0].shape[1] + next(iter(dataloader))[1].shape[1]
    model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)

    # Entraîner le modèle
    loss = train_vae(model, dataloader, lr, device=device, epochs=10)
    return loss

def run_optuna(n_trials=30, save_best=True):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    if save_best:
        import json
        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
    return best_params
