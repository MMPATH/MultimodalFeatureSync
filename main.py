import argparse
import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import glob
import gc
from tqdm import tqdm
from video_preprocessing import process_video
from utilities import (
    convert_mp4_to_wav,
    rolling_window,
    seconds_to_samples,
    resample_vocal_features,
    resample_facial_features,
    resample_linguistic_features,
)
from diarize import load_and_diarize_audio, keep_ranges
from extract_features import (
    extract_acoustic_features,
    extract_linguistic_features,
    extract_facial_features,
)
from config import (
    audio_dir,
    video_dir,
    acoustic_dir,
    facial_dir,
    linguistic_dir,
    use_auth_token,
    of_bin_loc,
)


def process_file_diarization(file_path):
    """
    Process a single file through the diarization part of the pipeline.

    Args:
    file_path (str): Path to the file to process.
    """
    # Assume file_path is a path to a video file
    base_fname = os.path.splitext(os.path.basename(file_path))[0]

    # Step 1: Video Preprocessing
    print(f"Processing video file: {file_path}")
    process_video(file_path, video_dir)

    # Step 2: Audio Extraction
    print(f"Extracting audio from video file: {file_path}")
    audio_file = f"{base_fname}.wav"
    # Check if audio file already exists
    if not os.path.exists(os.path.join(audio_dir, audio_file)):
        convert_mp4_to_wav(file_path, os.path.join(audio_dir, audio_file))
    else:
        print("Audio file already exists, skipping conversion.")

    # Step 3: Audio Processing
    s1_write_path = os.path.join(audio_dir, f"{base_fname}_s1.wav")
    s2_write_path = os.path.join(audio_dir, f"{base_fname}_s2.wav")

    # Check if diarized audio files already exist
    if os.path.exists(s1_write_path) and os.path.exists(s2_write_path):
        print("Diarized audio files already exist, skipping diarization.")
        return

    print(f"Processing audio file: {audio_file}")
    signal, R1, R2 = load_and_diarize_audio(base_fname, audio_dir, use_auth_token)
    s1_signal = keep_ranges(signal, R1)
    s2_signal = keep_ranges(signal, R2)

    # Save the diarized signals
    sf.write(s1_write_path, s1_signal, 16000)
    sf.write(s2_write_path, s2_signal, 16000)


def process_file_features(file_path, speaker_label):
    """
    Process a single file through the feature extraction part of the pipeline.

    Args:
    file_path (str): Path to the file to process.
    speaker_label (str): Label of the speaker to process ('s1' or 's2').
    """
    base_fname = os.path.splitext(os.path.basename(file_path))[0]
    audio_file = os.path.join(audio_dir, f"{base_fname}_{speaker_label}.wav")

    # Load the selected speaker's signal
    signal = librosa.load(audio_file, sr=16000)[0]

    # Step 4: Feature Extraction
    # Check for existing feature files
    skip_acoustic = False
    skip_linguistic = False
    skip_facial = False
    if os.path.exists(os.path.join(acoustic_dir, f"{base_fname}.csv")):
        print("Features already extracted for this file, skipping acoustic.")
        skip_acoustic = True
    if os.path.exists(os.path.join(linguistic_dir, f"{base_fname}.npy")):
        print("Features already extracted for this file, skipping linguistic.")
        skip_linguistic = True
    if os.path.exists(os.path.join(facial_dir, f"{base_fname}.csv")):
        print("Features already extracted for this file, skipping facial.")
        skip_facial = True
    # Create rolling window
    window_size = seconds_to_samples(5)
    step_size = seconds_to_samples(1)
    rolling_signal_window = rolling_window(signal, window_size, step_size)
    if not skip_acoustic:
        print(f"Extracting acoustic features from audio file: {audio_file}")
        extract_acoustic_features(signal, base_fname)
    if not skip_linguistic:
        print(f"Extracting linguistic features from audio file: {audio_file}")
        extract_linguistic_features(rolling_signal_window, use_auth_token, base_fname)
    if not skip_facial:
        print(f"Extracting facial features from video file: {file_path}")
        extract_facial_features(of_bin_loc, video_dir, base_fname)

    # Step 5: Data Resampling
    print("Resampling features")
    # Read the feature files
    facial_features_path = os.path.join(facial_dir, f"{base_fname}.csv")
    vocal_features_path = os.path.join(acoustic_dir, f"{base_fname}.csv")
    linguistic_features_path = os.path.join(linguistic_dir, f"{base_fname}.npy")

    facial_features = pd.read_csv(facial_features_path)
    facial_features = facial_features.drop(
        ["frame", "face_id", "timestamp"], axis=1
    )  # Clean facial features
    vocal_features = pd.read_csv(vocal_features_path)
    linguistic_features = np.load(linguistic_features_path)

    # Resample each feature type
    vocal_features_resampled = resample_vocal_features(vocal_features)
    facial_features_resampled = resample_facial_features(
        facial_features, vocal_features_resampled.index
    )
    linguistic_features_resampled = resample_linguistic_features(linguistic_features)

    # Step 6: Feature Combination
    print("Combining features")
    combined_features = pd.concat(
        [
            facial_features_resampled,
            vocal_features_resampled,
            linguistic_features_resampled,
        ],
        axis=1,
    )
    combined_features = combined_features.dropna()
    combined_features.to_csv(os.path.join("./features/combined", f"{base_fname}.csv"))

    # Explicitly delete large objects and run garbage collection to free up memory
    del signal, facial_features, vocal_features, linguistic_features
    del vocal_features_resampled, facial_features_resampled, linguistic_features_resampled
    del combined_features
    gc.collect()

import csv


def read_speaker_labels(csv_path, identity="patient"):
    """
    Read the CSV file and return a dictionary mapping file names to patient speaker labels.

    Args:
    csv_path (str): Path to the CSV file.

    Returns:
    dict: A dictionary mapping file names to speaker labels ('s1' or 's2').
    """
    speaker_labels = {}
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_name_base = row["File Name (Base)"]
            s1_speaker = row["s1 Speaker Identity"]
            speaker_labels[file_name_base] = "s1" if s1_speaker == identity else "s2"
    return speaker_labels


def main():
    parser = argparse.ArgumentParser(
        description="Process video files for diarization / feature extraction."
    )
    parser.add_argument(
        "input_path", help="Path to the input directory containing MP4 files"
    )
    parser.add_argument(
        "--csv_path", help="Optional: Path to the CSV file containing speaker labels", default=None
    )
    parser.add_argument(
        "--stage",
        choices=["diarization", "features"],
        help="Processing stage: 'diarization' or 'features'"
    )

    args = parser.parse_args()

    # Get all MP4 files in the directory
    file_paths = glob.glob(os.path.join(args.input_path, "*.mp4"))

    if args.stage == "diarization":
        for file_path in tqdm(file_paths, desc="Processing files"):
            process_file_diarization(file_path)

    elif args.stage == "features":
        if args.csv_path is None:
            print("CSV path is required for feature extraction stage.")
            return

        # Read speaker labels from the CSV file
        speaker_labels = read_speaker_labels(args.csv_path)

        for file_path in tqdm(file_paths, desc="Processing files"):
            base_fname = os.path.splitext(os.path.basename(file_path))[0]
            speaker_a = speaker_labels.get(base_fname, None)
            if speaker_a:
                process_file_features(file_path, speaker_a)
            else:
                print(f"Speaker label for {base_fname} not found in CSV.")
    else:
        print("Invalid stage argument. Please choose 'diarization' or 'features'.")

if __name__ == "__main__":
    main()

