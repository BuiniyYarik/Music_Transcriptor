import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from numpy.lib import stride_tricks


def calculate_cqt_stats(data_dir, frame_size=2048, overlap_fac=0.5, bins_per_octave=12, n_octaves=7, data_size=10):
    """
    Calculates the mean and standard deviation of the CQT of the dataset.

    Parameters:
    data_dir (str): Path to the dataset folder
    frameSize (int): Frame size for the CQT
    overlapFac (float): Overlap factor for the CQT
    bins_per_octave (int): Number of bins per octave for the CQT
    n_octaves (int): Number of octaves for the CQT
    data_size (int): Size of the CQT dataset

    Returns:
    tuple: Mean and standard deviation of the CQT
    """
    sum = None
    sum_of_squares = None
    count = 0
    num_file_processed = 0

    for file in tqdm(os.listdir(data_dir), desc=f"Calculating CQT stats", total=data_size):
        if num_file_processed == data_size:
            break
        if file.endswith('.wav'):
            cqt = generate_cqt(os.path.join(data_dir, file), frame_size, overlap_fac, bins_per_octave, n_octaves)
            if sum is None:
                sum = np.sum(cqt, axis=0)
                sum_of_squares = np.sum(cqt ** 2, axis=0)
            else:
                sum += np.sum(cqt, axis=0)
                sum_of_squares += np.sum(cqt ** 2, axis=0)
            count += cqt.shape[0]
            num_file_processed += 1

    mean = sum / count
    variance = (sum_of_squares - (sum ** 2) / count) / count
    std_dev = np.sqrt(variance)
    return mean, std_dev


def calculate_stft_stats(data_dir, frame_size=2048, overlap_fac=0.5, data_size=10):
    """
    Calculates the mean and standard deviation of the STFT of the dataset.

    Parameters:
    data_dir (str): Path to the dataset folder
    frame_size (int): Frame size for the STFT
    overlap_fac (float): Overlap factor for the STFT
    data_size (int): Size of the STFT dataset

    Returns:
    tuple: Mean and standard deviation of the STFT
    """
    sum_ = None
    sum_of_squares = None
    count = 0
    num_file_processed = 0

    # Wrap the file iteration with tqdm for a progress bar
    for file in tqdm(os.listdir(data_dir), desc=f"Calculating STFT stats", total=data_size):
        if num_file_processed == data_size:
            break
        if file.endswith('.wav'):
            sfft = generate_stft(os.path.join(data_dir, file), frame_size, overlap_fac)
            if sum_ is None:
                sum_ = np.sum(sfft, axis=0)
                sum_of_squares = np.sum(sfft ** 2, axis=0)
            else:
                sum_ += np.sum(sfft, axis=0)
                sum_of_squares += np.sum(sfft ** 2, axis=0)
            count += sfft.shape[0]
            num_file_processed += 1

    mean = sum_ / count
    variance = (sum_of_squares - (sum_ ** 2) / count) / count
    std_dev = np.sqrt(variance)

    return mean, std_dev


def generate_cqt(audio_file, frame_size=2048, overlap_fac=0.5, bins_per_octave=12, n_octaves=7, mean=None, std=None):
    """
    Generates the CQT of an audio file.

    Parameters:
    audio_file (str): Path to the audio file
    frame_size (int): Frame size for the CQT
    overlap_fac (float): Overlap factor for the CQT
    bins_per_octave (int): Number of bins per octave for the CQT
    n_octaves (int): Number of octaves for the CQT
    mean (np.array): Mean of the CQT of the dataset
    std (np.array): Standard deviation of the CQT of the dataset

    Returns:
    np.array: CQT of the audio file (spectrogram)
    """
    y, sr = librosa.load(audio_file, sr=44100)
    hop_length = int(frame_size - (overlap_fac * frame_size))

    C = librosa.cqt(y, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_octaves*bins_per_octave)
    C_dB = librosa.amplitude_to_db(abs(C))

    if mean is not None and std is not None:
        mean = mean.reshape(-1, 1)
        std = std.reshape(-1, 1)
        C_dB = (C_dB - mean) / std

    return C_dB.T


def generate_stft(audio_file, frame_size=2048, overlap_fac=0.5, window=np.hanning, mean=None, std=None):
    """
    Generates the STFT of an audio file.

    Parameters:
    audio_file (str): Path to the audio file
    frame_size (int): Frame size for the STFT
    overlap_fac (float): Overlap factor for the STFT
    window (function): Window function for the STFT
    mean (np.array): Mean of the STFT of the dataset
    std (np.array): Standard deviation of the STFT of the dataset

    Returns:
    np.array: STFT of the audio file (spectrogram)
    """
    sig, sr = librosa.load(audio_file, sr=44100)
    win = window(frame_size)
    hopSize = int(frame_size - np.floor(overlap_fac * frame_size))

    samples = np.append(np.zeros(np.int64(np.floor(frame_size/2.0))), sig)
    cols = np.int64(np.ceil((len(samples) - frame_size) / float(hopSize)) + 1)
    samples = np.append(samples, np.zeros(frame_size))

    frames = stride_tricks.as_strided(
        samples, shape=(cols, frame_size), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    stft = np.fft.rfft(frames)

    # Apply standardization if mean and std are provided
    if mean is not None and std is not None:
        mean = mean.reshape(1, -1)  # Reshape mean to be broadcastable
        std = std.reshape(1, -1)    # Reshape std to be broadcastable
        stft = (stft - mean) / std

    return np.real(np.abs(stft))


def log_scale_spec(spec, sr=44100, factor=20.):
    """
    Scale a spectrogram to log scale (dB).

    Parameters:
    spec (np.array): Spectrogram to be scaled
    sr (int): Sampling rate
    factor (float): Scaling factor

    Returns:
    np.array: Scaled spectrogram
    """
    time_bins, freq_bins = np.shape(spec)

    scale = np.linspace(0, 1, freq_bins) ** factor
    scale *= (freq_bins - 1) / max(scale)
    scale = np.unique(np.round(scale)).astype(int)  # Ensure indices are integers

    newspec = np.complex128(np.zeros([time_bins, len(scale)]))
    for i in range(len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i+1]], axis=1)

    allfreqs = np.abs(np.fft.fftfreq(freq_bins*2, 1./sr)[:freq_bins+1])
    freqs = [np.mean(allfreqs[scale[i]:scale[i+1]]) for i in range(len(scale)-1)] + [np.mean(allfreqs[scale[-1]:])]

    return newspec, freqs


def generate_note_labels(label_file, n_frames, hop_length, min_note_value=21):
    """
    Generates the labels represented played notes for an audio file.

    Parameters:
    label_file (str): Path to the label file
    n_frames (int): Number of frames in the audio file
    hop_length (int): Hop length (stride) for the CQT
    min_note_value (int): Minimum MIDI note value

    Returns:
    np.array: Labels for the audio file
    """
    labels_df = pd.read_csv(label_file)
    labels = np.zeros((n_frames, 88), dtype=int)

    for _, row in labels_df.iterrows():
        start_frame = int(row['start_time'] // hop_length)
        end_frame = int(row['end_time'] // hop_length)
        note = int(row['note']) - min_note_value
        if 0 <= note < 88:
            labels[start_frame:end_frame, note] = 1

    return labels


def generate_note_with_instrument_labels(label_file, n_frames, hop_length, min_note_value=21, n_instruments=11):
    """
    Generates the labels represented played notes with instrument types for an audio file.

    Parameters:
    label_file (str): Path to the label file
    n_frames (int): Number of frames in the audio file
    hop_length (int): Hop length (stride) for the CQT
    min_note_value (int): Minimum MIDI note value
    n_instruments (int): Number of instrument types

    Returns:
    np.array: Labels for the audio file
    """

    instruments_mapping = {
        1: 0,
        7: 1,
        41: 2,
        42: 3,
        43: 4,
        44: 5,
        61: 6,
        69: 7,
        71: 8,
        72: 9,
        74: 10
    }

    labels_df = pd.read_csv(label_file)
    labels = np.zeros((n_frames, 88 * (n_instruments + 1)), dtype=int)  # +1 for the 'not played' condition

    for _, row in labels_df.iterrows():
        start_frame = int(row['start_time'] // hop_length)
        end_frame = int(row['end_time'] // hop_length)
        note = int(row['note']) - min_note_value
        instrument = instruments_mapping.get(int(row['instrument']))

        if 0 <= note < 88:
            instrument_idx = note * (n_instruments + 1) + instrument
            labels[start_frame:end_frame, instrument_idx] = 1

    return labels


# Example usage:
# mean, std_dev = calculate_cqt_stats(train_data_dir)
# X_train, y_train = process_files(train_data_dir, train_label_dir, generate_note_labels, mean=mean, std=std_dev)
def process_files_using_cqt(data_dir, label_dir, preprocess_labels,
                            frame_size=2048, overlap_fac=0.5,
                            bins_per_octave=12, n_octaves=7,
                            sr=44100,
                            mean=None, std=None,
                            data_size=10):
    """
    Processes the audio files in the dataset using the Constant-Q Transform (CQT) and generates the labels

    Parameters:
    data_dir (str): Path to the dataset folder
    label_dir (str): Path to the label folder
    preprocess_labels (function): Function to preprocess the labels
    frame_size (int): Frame size for the CQT
    overlap_fac (float): Overlap factor for the CQT
    bins_per_octave (int): Number of bins per octave
    n_octaves (int): Number of octaves
    hop_length (int): Hop length (stride) for the CQT
    sr (int): Sampling rate of the audio files
    mean (np.array): Mean of the CQT of the dataset
    std (np.array): Standard deviation of the CQT of the dataset
    data_size (int): Size of the CQT dataset

    Returns:
    tuple: Lists of CQTs and labels
    """
    X, y = [], []
    hop_length = int(frame_size - (overlap_fac * frame_size))
    num_file_processed = 0

    for file in tqdm(os.listdir(data_dir), desc=f"Processing files in '{data_dir}' and '{label_dir}'", total=data_size):
        if num_file_processed == data_size:
            break
        if file.endswith('.wav'):
            audio_processing = generate_cqt(os.path.join(data_dir, file), frame_size, overlap_fac, bins_per_octave,
                                            n_octaves, mean, std)
            n_frames = audio_processing.shape[0]
            label_file = os.path.join(label_dir, file.replace('.wav', '.csv'))
            labels = preprocess_labels(label_file, n_frames, hop_length)
            X.append(audio_processing)
            y.append(labels)

            num_file_processed += 1

    return X, y


# Example usage:
# mean, std_dev = calculate_sfft_stats(train_data_dir)
# X_train, y_train = process_files(train_data_dir, train_label_dir, generate_note_labels, mean=mean, std=std_dev)
def process_files_using_stft(data_dir, label_dir, preprocess_labels,
                             frame_size=2048, overlap_fac=0.5,
                             sr=44100, mean=None, std=None, data_size=10):
    """
    Processes the audio files in the dataset using the Short-Time Fourier Transform (STFT) and generates the labels

    Parameters:
    data_dir (str): Path to the dataset folder
    label_dir (str): Path to the label folder
    preprocess_labels (function): Function to preprocess the labels
    frame_size (int): Frame size for the STFT
    overlap_fac (float): Overlap factor for the STFT
    sr (int): Sampling rate of the audio files
    mean (np.array): Mean of the STFT of the dataset
    std (np.array): Standard deviation of the STFT of the dataset
    data_size (int): Size of the STFT dataset

    Returns:
    tuple: Lists of STFTs and labels
    """

    X, y = [], []
    hop_length = int(frame_size - (overlap_fac * frame_size))
    num_file_processed = 0

    for file in tqdm(os.listdir(data_dir), desc=f"Processing files in '{data_dir}' and '{label_dir}'", total=data_size):
        if num_file_processed == data_size:
            break
        if file.endswith('.wav'):
            stft_data = generate_stft(os.path.join(data_dir, file), frame_size, overlap_fac, mean=mean, std=std)
            log_spec, freqs = log_scale_spec(stft_data, sr=sr)

            n_frames = log_spec.shape[0]
            label_file = os.path.join(label_dir, file.replace('.wav', '.csv'))
            labels = preprocess_labels(label_file, n_frames, hop_length)

            X.append(log_spec)
            y.append(labels)

            num_file_processed += 1

    return X, y
