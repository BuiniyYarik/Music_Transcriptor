import numpy as np
import librosa
import os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


# Generator class
class DataGenerator(Sequence):
    def __init__(self, batch_size=32, dim=(5, 2048), n_channels=1, sampling_rate=44100, stride=512,
                 n_classes=88, shuffle=True, transform_type='fft', file_path="musicnet.npz"):
        """
        Initialization
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating data dimensionality
        :param n_channels: number of channels
        :param sampling_rate: sampling rate
        :param stride: stride for the transformation
        :param n_classes: number of classes
        :param shuffle: shuffle label indices after each epoch
        :param transform_type: type of transformation to apply to the data
        :param file_path: path to the data file
        """
        self.dim = dim
        self.batch_size = batch_size
        self.file_path = file_path
        self.n_channels = n_channels
        self.fs = sampling_rate
        self.stride = stride
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.transform_type = transform_type
        self.on_epoch_end()
        self.features, self.labels = self.load_data()
        self.indexes = np.arange(len(self.features))
        self.max_value = self._find_max_value()
        self.instruments_mapping = {1: 0, 7: 1, 41: 2, 42: 3, 43: 4, 44: 5, 61: 6, 69: 7, 71: 8, 72: 9, 74: 10}
        # self.num_windows = self.fs / self.stride

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: The number of batches per epoch
        """
        return int(np.floor(len(self.features) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self._data_generation(list_IDs_temp)

        return X, y

    def load_data(self):
        """
        Load the data
        :return: features and labels
        """
        data = np.load(self.file_path, allow_pickle=True, encoding='latin1')
        X = {}
        y = {}
        for file in data.files:
            X[file] = data[file][0]
            intervals = data[file][1]
            y[file] = np.zeros((len(intervals), 4))
            for i, interval in enumerate(intervals):
                y[file][i][0] = interval[0]
                y[file][i][1] = interval[1]
                y[file][i][2] = self.instruments_mapping[interval[2][0]]
                y[file][i][3] = interval[2][1]
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            signal = self.features[ID]

            # Transform and normalize
            f_transform = self._transform_audio(signal)

            # Normalize
            f_transform /= self.max_value

            X[i, ] = f_transform.reshape(*self.dim, self.n_channels)

            # Store class
            y[i] = self.labels[ID]

        return X, to_categorical(y, num_classes=self.n_classes)

    def _transform_audio(self, signal):
        if self.transform_type == 'fft':
            return np.abs(np.fft.fft(signal, n=self.dim[1]))
        elif self.transform_type == 'cqt':
            f_transform = librosa.cqt(signal, sr=44100, n_bins=self.dim[1], bins_per_octave=48,
                                      fmin=librosa.note_to_hz('A1'), norm=1)
            return np.abs(f_transform)
        else:
            raise ValueError("Invalid transformation type")

    def _process_label(self, interval, total_windows, num_instruments=11):
        """
        Encodes labels into one-hot format for both notes and instruments.
        :param interval: The interval data containing start time, end time, note index, and instrument ID.
        :param total_windows: Total number of windows in the song.
        :param num_instruments: Total number of possible instruments.
        :return: One-hot encoded matrix for notes and instruments.
        """
        # Initialize the one-hot matrix for notes (88) and instruments (11)
        one_hot_matrix = np.zeros((total_windows, self.n_classes + num_instruments))

        # Extract the start time, end time, note index, and instrument ID from the interval
        start_time, end_time, note_index, instrument_id = interval

        # Calculate the range of windows for the note
        start_window = int(start_time * self.fs / self.stride)
        end_window = int(end_time * self.fs / self.stride)

        # Set the one-hot encoding for the note index
        one_hot_matrix[start_window:end_window, note_index] = 1

        # Set the one-hot encoding for the instrument ID
        one_hot_matrix[:, self.n_classes + instrument_id] = 1  # Assuming instrument_id is 0-indexed

        return one_hot_matrix

    def _find_max_value(self):
        max_value = 0
        for X in self.features:
            # Apply transformation
            if self.transform_type == 'fft':
                transformed_data = np.abs(np.fft.fft(X, n=self.dim[1]))
            elif self.transform_type == 'cqt':
                transformed_data = np.abs(librosa.cqt(X, sr=44100, n_bins=2048, bins_per_octave=48,
                                                      fmin=librosa.note_to_hz('A1')))
            max_value = max(max_value, np.max(transformed_data))
        return max_value
