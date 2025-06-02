import numpy as np
from .normalize import standardize
from torch.utils.data import DataLoader, TensorDataset

def load_normalize_data(batch_size=64, shuffle=True, return_dataset=False):
    """
    Charge les données, les standardise, et retourne soit un DataLoader, soit le dataset complet.
    """
    # Chargement des données
    X_data = np.load('data/X_data_array_5000.npy')
    y_data = np.load('data/y_data_array_5000.npy')

    # Standardisation
    X_scaled, y_scaled, scaler_x, scaler_y = standardize(X_data, y_data)

    # Création du dataset
    dataset = TensorDataset(X_scaled, y_scaled)

    if return_dataset:
        return dataset, scaler_x, scaler_y
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        print("Data loaded and normalized.")
        return dataloader, scaler_x, scaler_y
