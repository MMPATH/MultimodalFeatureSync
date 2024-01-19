import librosa
import os
import numpy as np
import pandas as pd
import subprocess
from numpy.lib.stride_tricks import sliding_window_view


def seconds_to_samples(seconds, sample_rate=16000):
    """Convert seconds to samples"""
    return int(seconds * sample_rate)


def samples_to_seconds(samples, sample_rate=16000):
    """Convert samples to seconds"""
    return samples / sample_rate


def load_wav_file(filename, directory, sample_rate=16000):
    """Load a .wav file into memory"""
    signal, _ = librosa.load(os.path.join(directory, filename), sr=sample_rate)
    return signal


def rolling_window(signal, window_size, step_size):
    """Given a 1D or 2D signal, return a rolling window of that signal
    with a specified window size and step size.

    2D signals are assumed to be of shape (samples, features)
    and will be returned as (windows, samples, features)]"""

    # Convert signal to numpy array
    np.asarray(signal)

    # Check if signal is 1D or 2D and return rolling window
    if len(signal.shape) == 1:
        return sliding_window_view(signal,
                                   window_shape=window_size)[::step_size]

    if len(signal.shape) == 2:
        n_features = signal.shape[1]
        return sliding_window_view(signal,
                                   window_shape=(window_size, n_features))[
            ::step_size
        ].squeeze()


def convert_mp4_to_wav(inputfile, outputfile, sample_rate=16000):
    """
    Convert a single MP4 file to WAV format.

    Args:
    inputfile (str): Path to the input MP4 file.
    outputfile (str): Path for the output WAV file.
    sample_rate (int): Sample rate for the WAV file.

    Returns:
    None
    """
    try:
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                inputfile,
                "-vn",
                "-ar",
                str(sample_rate),
                "-f",
                "wav",
                outputfile,
            ]
        )
    except Exception:
        print("Error converting file: {inputfile}")


def round_to_base(x, base=0.01):
    """
    Rounds a number to the nearest specified base.

    Args:
    x (float): The number to be rounded.
    base (float, optional): The base to round to. Defaults to 0.01.

    Returns:
    float: The number rounded to the nearest specified base.
    """
    return round(base * round(float(x) / base), 2)


def resample_facial_features(facial_features, target_index):
    """
    Resamples facial features to align with a target index based on a fixed sr.

    Args:
    facial_features (pandas.DataFrame): DataFrame containing facial features.
    target_index (pandas.Index): The target index to align the facial features.

    Returns:
    pandas.DataFrame: The resampled facial features DataFrame.
    """
    facial_features.index = (facial_features.index / 25).map(
        round_to_base
    )  # facial_sr = 25 Hz
    return resample_features(facial_features, target_index)


def resample_vocal_features(vocal_features):
    """
    Resamples vocal features to a fixed sample rate.

    Args:
    vocal_features (pandas.DataFrame): DataFrame containing vocal features.

    Returns:
    pandas.DataFrame: The vocal features DataFrame with resampled index.
    """
    vocal_features["time"] = vocal_features.index / 100  # vocal_sr = 100 Hz
    vocal_features = vocal_features.set_index("time")
    vocal_features.index = vocal_features.index.map(round_to_base)
    return vocal_features


def resample_linguistic_features(linguistic_features):
    """
    Resamples linguistic features to align with a target index.

    Args:
    linguistic_features (numpy.ndarray): Array of linguistic features.
    target_index (pandas.Index): The target index to align the linguistic
                                 features with.

    Returns:
    pandas.DataFrame: The resampled linguistic features DataFrame.
    """
    linguistic_features_df = pd.DataFrame(linguistic_features)
    rounded_time_index = np.arange(
        0, len(linguistic_features), 1 / 100
    )  # assumes window step size of 1 second
    linguistic_features_resampled = linguistic_features_df.reindex(
        rounded_time_index, method="nearest"
    )
    linguistic_features_resampled.index = \
        linguistic_features_resampled.index.map(round_to_base)

    # Trim DataFrame as required (based on linguistic_window)
    linguistic_start_time = 2.5  # Half of the linguistic_window of 5 seconds
    linguistic_end_time = len(linguistic_features_resampled) - 2.5
    return linguistic_features_resampled.loc[linguistic_start_time:
                                             linguistic_end_time]


def resample_features(features, target_index, method="linear"):
    """
    Resamples a set of features to a target index using a specified
    interpolation method.

    Args:
    features (pandas.DataFrame): DataFrame containing features to be resampled.
    target_index (pandas.Index): The target index to resample the features to.
    method (str, optional): The interpolation method to use.

    Returns:
    pandas.DataFrame: The resampled features DataFrame.
    """
    features = features.reindex(target_index)
    return features.interpolate(method=method)
