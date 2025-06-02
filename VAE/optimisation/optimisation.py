import optuna
import torch
from VAE.vae import VAE  # ton module contenant Encoder, Decoder, VAE
from loss.loss import vae_loss  # fonction de perte VAE
from data_processing.dataload import load_normalize_data
from torch.utils.data import random_split, DataLoader
import optuna.visualization as vis
import plotly
print(plotly.__version__)

# ============================================================
# Nouvelle version de la fonction objective pour Optuna
# ============================================================

def train_vae(model, dataloader_train, dataloader_test, lr, device="cpu", epochs=10, trial=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    total_loss = 0

    # Paramètres de stagnation (early stopping personnalisé)
    patience = 200
    min_delta = 1e-4
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()

        for x_batch, y_batch in dataloader_train:
            xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
            recon, mu, log_var = model(xy_batch)
            loss = vae_loss(recon, xy_batch, mu, log_var)

            if torch.isnan(loss) or torch.isinf(loss):
                raise optuna.TrialPruned()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in dataloader_test:
                xy_val = torch.cat([x_val, y_val], dim=1).to(device)
                recon, mu, log_var = model(xy_val)
                loss = vae_loss(recon, xy_val, mu, log_var)
                val_loss += loss.item()
        val_loss /= len(dataloader_test)

        # Reporter la loss à Optuna
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping personnalisé (stagnation)
        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Pruned for stagnation at epoch {epoch+1}.")
            raise optuna.TrialPruned()

        print(f"Epoch {epoch+1}/{epochs}, loss: {val_loss:.4f}")

    return best_loss

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
    dataset, _, _ = load_normalize_data(batch_size=None, return_dataset=True)  # Adapter la fonction pour retourner le dataset complet

    # Split du dataset en train et test (80%/20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(42)  # Pour la reproductibilité
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=64, shuffle=False)


    batch = next(iter(dataloader_train))
    input_dim = batch[0].shape[1] + batch[1].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for Optuna:", device)

    # Créer modèle
    model = VAE(input_dim=input_dim, latent_dim=latent_dim,
                hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Entraîner
    loss = train_vae(model, dataloader_train, dataloader_test, lr=lr, device=device, epochs=2000, trial=trial)
    return loss

# ============================================================
# Fonction pour lancer l'optimisation Optuna
# ============================================================

def run_optuna(n_trials=30, save_best=True):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=25,       # nombre minimal d'étapes (par ex. epochs) avant de pouvoir être élagué
            max_resource=2000,     # nombre maximal d'étapes (epochs que tu passes à `train_vae`)
            reduction_factor=4    # facteur de réduction (standard = 3)
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

