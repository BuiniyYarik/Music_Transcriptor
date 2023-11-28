import numpy as np
import pandas as pd
import h5py
import os


def find_note_range(file_paths):
    """
    Finds the minimum and maximum note in the dataset.

    Parameters:
    file_paths (list): List of paths to the dataset folders (train and test)

    Returns:
    tuple: Minimum and maximum note in the dataset
    """
    min_note, max_note = float('inf'), 0
    for file_path in file_paths:
        for file in os.listdir(file_path):
            if file.endswith('.csv'):
                labels_df = pd.read_csv(os.path.join(file_path, file))
                min_note = min(min_note, labels_df['note'].min())
                max_note = max(max_note, labels_df['note'].max())
    return min_note, max_note + 1


def spectrogram_row_to_image(data, row_index, num_rows=5):
    """
    Transforms a specified spectrogram row into a multi-row image by combining it with its neighboring rows:
    1) row - current row (window) to be transformed
    2) row -> [row - (num_rows-1)/2, ..., row, ..., row + (num_rows-1)/2]

    Parameters:
    data (np.array): Input data (2D array where each row is a time window of the spectrogram)
    row_index (int): Index of the row to be transformed
    num_rows (int): Total number of rows in the resulting image (including the target row and its neighbors)

    Returns:
    np.array: Transformed image (2D array)
    """
    if num_rows % 2 == 0:
        raise ValueError("num_rows must be an odd number.")

    middle_index = num_rows // 2
    num_columns = data.shape[1]  # Number of columns in the input data
    transformed_row = np.zeros((num_rows, num_columns))

    for i in range(num_rows):
        current_row_index = row_index + i - middle_index
        if 0 <= current_row_index < len(data):
            transformed_row[i, :] = data[current_row_index]
        else:
            # If the current_row_index is out of bounds, it remains filled with zeros.
            pass

    return transformed_row


def find_max_length(data_list):
    """
    Finds the maximum length of the data arrays.

    Parameters:
    data_list (list): List of data arrays

    Returns:
    int: Maximum length of the data arrays
    """
    return max(data.shape[0] for data in data_list)


def pad_data(data, max_length, return_type=np.float32):
    """
    Pads the data arrays to the maximum length.

    Parameters:
    data (list): List of data arrays
    max_length (int): Maximum length of the data arrays

    Returns:
    np.array: Padded data arrays
    """
    padded_data = [np.pad(x, ((0, max_length - x.shape[0]), (0, 0)), 'constant') for x in data]
    return np.array(padded_data, dtype=return_type)


# Example usage for saving data:
# save_preprocessed_data(X_train, y_train, 'data/processed', 'train_data.npz')
def save_preprocessed_data(X, y, path, filename):
    """
    Saves preprocessed data (features and labels) to an HDF5 file.

    Parameters:
    X (np.array): Features
    y (np.array): Labels
    path (str): Path to the file
    filename (str): Name of the file

    Returns:
    None
    """
    print("Starting to save preprocessed data...")
    with h5py.File(os.path.join(path, filename), 'w') as h5f:
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('y', data=y)
    print(f"Data successfully saved to {os.path.join(path, filename)}")


# Example usage for loading data:
# X_train_loaded, y_train_loaded = load_preprocessed_data('data/processed', 'train_data.npz')
def load_preprocessed_data(path, filename, size=None):
    """
    Loads preprocessed data (features and labels) from an HDF5 file.

    Parameters:
    path (str): Path to the file
    filename (str): Name of the file
    size (int): Number of data points to load (None for all data)

    Returns:
    Tuple[np.array, np.array]: Features and labels
    """
    with h5py.File(os.path.join(path, filename), 'r') as h5f:
        X = h5f['X'][:size]
        y = h5f['y'][:size]
    return X, y
