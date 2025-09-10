import h5py
import numpy as np

def load_h5_data(train_x_path, train_y_path, val_x_path, val_y_path):
    """
    Load train and validation datasets from HDF5 files.
    """
    # Load training data
    with h5py.File(train_x_path, "r") as f:
        X_train = np.array(f["x"])
    with h5py.File(train_y_path, "r") as f:
        y_train = np.array(f["y"])

    # Load validation data
    with h5py.File(val_x_path, "r") as f:
        X_val = np.array(f["x"])
    with h5py.File(val_y_path, "r") as f:
        y_val = np.array(f["y"])

    print("Train images:", X_train.shape, "Train labels:", y_train.shape)
    print("Validation images:", X_val.shape, "Validation labels:", y_val.shape)

    # Normalize images
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0

    return X_train, y_train, X_val, y_val
