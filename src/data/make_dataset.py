from torch.utils.data import Dataset
from src.preprocessing.data_utils import spectrogram_row_to_image
import h5py
import os
import torch


class AudioDataset(Dataset):
    def __init__(self, data_path, filename, num_rows=15):
        self.file_path = os.path.join(data_path, filename)
        self.num_rows = num_rows
        with h5py.File(self.file_path, 'r') as h5f:
            self.n_samples = h5f['X'].shape[0]

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as h5f:
            X_sample = spectrogram_row_to_image(h5f['X'], index, self.num_rows)
            X_sample = torch.tensor(X_sample, dtype=torch.float32)
            y_sample = torch.tensor(h5f['y'][index], dtype=torch.float32)

        return X_sample, y_sample

    def __len__(self):
        return self.n_samples
