import numpy as np
from .normalize import standardize
from torch.utils.data import DataLoader, TensorDataset

def load_normalize_data(batch_size=64, shuffle=True):
    
    X_data = np.load('data/X_data_array_5000.npy')
    y_data = np.load('data/y_data_array_5000.npy')

    X_scaled, y_scaled, scaler_x, scaler_y = standardize(X_data, y_data)

    dataset = TensorDataset(X_scaled, y_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print("Data loaded and normalized.")

    return dataloader, scaler_x, scaler_y
