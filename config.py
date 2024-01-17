# config.py
import os

# Base directories
data_dir = "./data/"
feature_dir = "./features/"

# Specific directories for different types of data and features
raw_video_dir = os.path.join(data_dir, "raw_video")
video_dir = os.path.join(data_dir, "video")
audio_dir = os.path.join(data_dir, "audio")
acoustic_dir = os.path.join(feature_dir, "acoustic")
facial_dir = os.path.join(feature_dir, "facial")
linguistic_dir = os.path.join(feature_dir, "linguistic")
combined_dir = os.path.join(feature_dir, "combined")

# Authentication token for pyannote
use_auth_token = use_auth_token = os.environ.get("PYANNOTE_AUTH_TOKEN", None)
# OpenFace binary location
of_bin_loc = os.environ.get("OPENFACE_BIN_LOC", None)


def ensure_directories_exist():
    """
    Ensure that all necessary directories exist.
    """
    for dir_path in [
        data_dir,
        feature_dir,
        raw_video_dir,
        video_dir,
        audio_dir,
        acoustic_dir,
        facial_dir,
        linguistic_dir,
        combined_dir,
    ]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


ensure_directories_exist()
