from sklearn.preprocessing import StandardScaler
import torch

def standardize(x, y):
    """
    Standardise les arrays x et y et retourne les tensors standardisés.
    """
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
    y_scaled = torch.tensor(y_scaled, dtype=torch.float32)

    return x_scaled, y_scaled, scaler_x, scaler_y


# def inverse_standardize(x_scaled, y_scaled, scaler_x, scaler_y):
#     """
#     Inverse standardise les tensors x_scaled et y_scaled en utilisant les scalers fournis.
#     """
#     x_inv = scaler_x.inverse_transform(x_scaled.numpy())
#     y_inv = scaler_y.inverse_transform(y_scaled.numpy())

#     x_inv = torch.tensor(x_inv, dtype=torch.float32)
#     y_inv = torch.tensor(y_inv, dtype=torch.float32)

#     return x_inv, y_inv