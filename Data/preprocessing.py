import numpy as np
import librosa
import pandas as pd


class DataPreprocessor:
    def __init__(self, dim=(5, 2048), sampling_rate=44100, stride=512,
                 n_notes=88, transform_type='fft', file_path="musicnet.npz", metadata_path="musicnet_metadata.csv"):
        """
        Initialization
        :param dim: tuple indicating data dimensionality
        :param n_channels: number of channels
        :param sampling_rate: sampling rate
        :param stride: stride for the transformation
        :param n_classes: number of classes
        :param transform_type: type of transformation to apply to the data
        :param file_path: path to the data file
        :param metadata_path: path to the metadata CSV file
        """
        self.dim = dim
        self.file_path = file_path
        self.metadata_path = metadata_path
        self.fs = sampling_rate
        self.stride = stride
        self.n_notes = n_notes
        self.transform_type = transform_type
        self.metadata = self.load_metadata()
        self.instruments_mapping = {1: 0, 7: 1, 41: 2, 42: 3, 43: 4, 44: 5, 61: 6, 69: 7, 71: 8, 72: 9, 74: 10}

    def load_metadata(self):
        """
        Load the metadata from a CSV file
        :return: A dictionary with file IDs as keys and durations in seconds as values
        """
        metadata_df = pd.read_csv(self.metadata_path)
        metadata = {row['id']: row['seconds'] for index, row in metadata_df.iterrows()}
        return metadata

    def load_data(self):
        """
        Load the data
        :return: features and labels
        """
        data = np.load(self.file_path, allow_pickle=True, encoding='latin1')
        size = 10
        X = np.empty(size, dtype=object)
        y = np.empty(size, dtype=object)
        for i, file_id_str in enumerate(data.files):
            if i == size:
                break
            file_id = int(file_id_str)
            print("Loading file {} of {}".format(i + 1, size))

            duration_in_seconds = self.metadata[file_id]
            total_windows = int(np.ceil((self.fs / self.stride) * duration_in_seconds))

            X[i] = self.transform_audio(data[file_id_str][0], total_windows)

            intervals = data[file_id_str][1]
            y[i] = np.zeros((total_windows, self.n_notes + len(self.instruments_mapping)))
            for interval in intervals:
                start_time = interval[0]
                end_time = interval[1]
                instrument_id = interval[2][0]
                note = interval[2][1]

                note_index = note - 21  # Assuming MIDI note numbers start at 21
                instrument_index = self.instruments_mapping[instrument_id]
                self.process_label(y[i], start_time, end_time, note_index, instrument_index, total_windows)
            y[i] = y[i].T

        # Normalize the data
        max_value = -np.inf
        for i in range(len(X)):
            max_value = max(max_value, np.max(X[i]))
        X = X / max_value
        return X, y

    def process_label(self, one_hot_matrix, start_time, end_time, note_index, instrument_index, total_windows):
        """
        Encodes labels into one-hot format for both notes and instruments.
        :param one_hot_matrix: The matrix to encode the labels into.
        :param start_time: The start time of the note.
        :param end_time: The end time of the note.
        :param note_index: The index of the note.
        :param instrument_index: The index of the instrument.
        :param total_windows: Total number of windows in the song.
        """
        start_window = int(start_time * self.fs / self.stride)
        end_window = int(end_time * self.fs / self.stride)
        one_hot_matrix[start_window:end_window, note_index] = 1
        one_hot_matrix[:, self.n_notes + instrument_index] = 1

    def transform_audio(self, signal, total_windows):
        if self.transform_type == 'fft':
            # Calculate the number of samples per window
            window_samples = int(self.fs / self.stride)

            # Initialize the transformed signal array
            transformed_signal = np.zeros((self.dim[1], total_windows))

            # Loop over the signal in windows and compute the FFT for each
            for i in range(total_windows):
                start_sample = i * window_samples
                end_sample = start_sample + window_samples
                windowed_signal = signal[start_sample:end_sample]
                transformed_signal[:, i] = np.abs(np.fft.fft(windowed_signal, n=self.dim[1]))

            return transformed_signal

        elif self.transform_type == 'cqt':
            # Compute the CQT for the entire signal
            cqt_result = np.abs(librosa.cqt(signal, sr=self.fs, hop_length=self.stride, n_bins=self.dim[1]))

            # If the CQT result has more frames than total_windows, truncate it
            if cqt_result.shape[1] > total_windows:
                cqt_result = cqt_result[:, :total_windows]

            # If the CQT result has fewer frames, pad it with zeros
            elif cqt_result.shape[1] < total_windows:
                padding = np.zeros((self.dim[1], total_windows - cqt_result.shape[1]))
                cqt_result = np.hstack((cqt_result, padding))

            return cqt_result

        else:
            raise ValueError("Invalid transformation type")

