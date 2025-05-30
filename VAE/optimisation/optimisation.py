import optuna
import torch
from VAE.vae import VAE  # ton module contenant Encoder, Decoder, VAE
from loss.loss import vae_loss  # fonction de perte VAE
from data_processing.dataload import load_normalize_data

# ============================================================
# Nouvelle version de la fonction objective pour Optuna
# ============================================================

def train_vae(model, dataloader, lr=1e-3, device="cpu", epochs=10, trial=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss = 0

    # Paramètres de stagnation (early stopping personnalisé)
    patience = 10
    min_delta = 1e-4
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0

        for x_batch, y_batch in dataloader:
            xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
            recon, mu, log_var = model(xy_batch)
            loss = vae_loss(recon, xy_batch, mu, log_var)

            if torch.isnan(loss) or torch.isinf(loss):
                raise optuna.TrialPruned()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        total_loss += epoch_loss

        # Reporter la loss à Optuna
        if trial is not None:
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping personnalisé (stagnation)
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Pruned for stagnation at epoch {epoch+1}.")
            raise optuna.TrialPruned()

        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

    avg_loss = total_loss / (len(dataloader) * epochs)
    return avg_loss

def objective(trial):
    # Paramètres d’architecture
    latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Paramètres d’optimisation
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Charger données
    dataloader, _, _ = load_normalize_data(batch_size=32)
    batch = next(iter(dataloader))
    input_dim = batch[0].shape[1] + batch[1].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for Optuna:", device)

    # Créer modèle
    model = VAE(input_dim=input_dim, latent_dim=latent_dim,
                hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Entraîner
    loss = train_vae(model, dataloader, lr=lr, device=device, epochs=200, trial=trial)
    return loss

# ============================================================
# Fonction pour lancer l'optimisation Optuna
# ============================================================

def run_optuna(n_trials=30, save_best=True):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=0,   # Laisse les 0 premiers essais se faire entièrement
            n_warmup_steps=10,    # Laisse les 10 premières époques d’un essai se dérouler sans pruning
            interval_steps=1      # Vérifie à chaque epoch après warmup
        )
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    if save_best:
        import json
        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
    return best_params

