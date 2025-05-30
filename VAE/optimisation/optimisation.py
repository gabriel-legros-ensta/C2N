import optuna
import torch
from VAE.vae import VAE  # ton module contenant Encoder, Decoder, VAE
from loss.loss import vae_loss  # fonction de perte VAE
from data_processing.dataload import load_normalize_data
import optuna.visualization as vis
import plotly
print(plotly.__version__)


# ============================================================
# Nouvelle version de la fonction objective pour Optuna
# ============================================================

def train_vae(model, dataloader, lr, device="cpu", epochs=10, trial=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss = 0

    # Paramètres de stagnation (early stopping personnalisé)
    patience = 50
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
    latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64])
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    print("Using parameters for Optuna:")
    print("Latent dimension:", latent_dim)  # Dimension latente     
    print("Hidden dimension:", hidden_dim)  # Dimension cachée
    print("Dropout rate:", dropout)  # Taux de dropout

    # Paramètres d’optimisation
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    print("Learning rate:", lr)  # Taux d'apprentissage

    # Charger données
    dataloader, _, _ = load_normalize_data(batch_size=64)
    batch = next(iter(dataloader))
    input_dim = batch[0].shape[1] + batch[1].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for Optuna:", device)

    # Créer modèle
    model = VAE(input_dim=input_dim, latent_dim=latent_dim,
                hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Entraîner
    loss = train_vae(model, dataloader, lr=lr, device=device, epochs=300, trial=trial)
    return loss

# ============================================================
# Fonction pour lancer l'optimisation Optuna
# ============================================================

def run_optuna(n_trials=30, save_best=True):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=0,
            n_warmup_steps=30,
            interval_steps=10
        )
    )

    # Valeurs initiales choisies par toi
    study.enqueue_trial({
        "latent_dim": 64,
        "hidden_dim": 512,
        "dropout": 0.0008850,
        "lr": 0.00011
    })

    # Tu peux en ajouter plusieurs si tu veux tester plusieurs configs de départ
    # study.enqueue_trial({
    # 'latent_dim': 64,
    # 'hidden_dim': 1024,
    # 'dropout': 0.0017742412691669177,
    # 'lr': 0.0005959581533257725,
    # })

    # study.enqueue_trial({...})

    study.optimize(objective, n_trials=n_trials)

        # Générer et afficher les graphiques d’optimisation
    # fig1 = vis.plot_optimization_history(study)
    # fig2 = vis.plot_param_importances(study)

    # fig1.show(renderer="browser")
    # fig2.show(renderer="browser")

    best_params = study.best_params
    if save_best:
        import json
        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
    return best_params

