import argparse
import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
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
from feature_combination import combine_features
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
    convert_mp4_to_wav(file_path, os.path.join(audio_dir, audio_file))

    # Step 3: Audio Processing
    print(f"Processing audio file: {audio_file}")
    signal, R1, R2 = load_and_diarize_audio(base_fname, audio_dir,
                                            use_auth_token)
    s1_signal = keep_ranges(signal, R1)
    s2_signal = keep_ranges(signal, R2)

    # Save the diarized signals
    sf.write(os.path.join(audio_dir, f"{base_fname}_s1.wav"), s1_signal, 16000)
    sf.write(os.path.join(audio_dir, f"{base_fname}_s2.wav"), s2_signal, 16000)


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
    # Create rolling window
    window_size = seconds_to_samples(5)
    step_size = seconds_to_samples(1)
    rolling_signal_window = rolling_window(signal, window_size, step_size)
    print(f"Extracting acoustic features from audio file: {audio_file}")
    extract_acoustic_features(signal, base_fname)
    print(f"Extracting linguistic features from audio file: {audio_file}")
    extract_linguistic_features(rolling_signal_window, use_auth_token,
                                base_fname)
    print(f"Extracting facial features from video file: {file_path}")
    extract_facial_features(of_bin_loc, video_dir, base_fname)

    # Step 5: Data Resampling
    print(f"Resampling features")
    # Read the feature files
    facial_features_path = os.path.join(facial_dir, f"{base_fname}.csv")
    vocal_features_path = os.path.join(acoustic_dir, f"{base_fname}.csv")
    linguistic_features_path = os.path.join(linguistic_dir,
                                            f"{base_fname}.npy")

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
    linguistic_features_resampled = resample_linguistic_features(
        linguistic_features)

    # Step 6: Feature Combination
    print(f"Combining features")
    combined_features = combine_features(
        facial_features_resampled,
        vocal_features_resampled,
        linguistic_features_resampled,
    )
    combined_features.to_csv(
        os.path.join("./features/combined", f"{base_fname}_combined.csv")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process video files for diarization / feature extraction."
    )
    parser.add_argument("input_path", help="Path to the input file")
    parser.add_argument(
        "--stage",
        choices=["diarization", "features"],
        help="Processing stage: 'diarization' or 'features'",
    )
    parser.add_argument(
        "--speaker",
        choices=["s1", "s2"],
        help="Speaker label for feature extraction stage ('s1' or 's2')",
    )

    args = parser.parse_args()

    if args.stage == "diarization":
        process_file_diarization(args.input_path)
    elif args.stage == "features" and args.speaker:
        process_file_features(args.input_path, args.speaker)
    else:
        print("Invalid arguments.")
        print("Please specify a processing stage and speaker label.")


if __name__ == "__main__":
    main()
