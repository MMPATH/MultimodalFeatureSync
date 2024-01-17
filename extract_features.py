import os
import torch
import opensmile
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from config import acoustic_dir, facial_dir, linguistic_dir
from transcribe import transcribe_rolling_window


def gen_sentence_embeddings(transcription_output):
    """
    Generates sentence embeddings for a list of transcribed text using BERT.

    This function tokenizes the input text, processes it through a BERT model,
    computes the average embedding from the second-to-last layer of the model.

    Args:
    transcription_output (list of str): A list of transcribed text strings.

    Returns:
    numpy.ndarray: An array of sentence embeddings, where each row corresponds
                   to the embedding of a sentence from the input list.

    Note:
    The function utilizes CUDA for GPU acceleration, assuming a CUDA-compatible
    environment is available.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True)

    model.to('cuda')
    model.eval()

    embeddings = []

    print("Generating sentence embeddings...")

    with torch.no_grad():
        for text in tqdm(transcription_output):
            # Tokenize and ensure the sequence is not longer than 512
            tokens = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512)
            tokens = {k: v.to('cuda') for k, v in tokens.items()}

            # Get hidden states
            outputs = model(**tokens)
            hidden_states = outputs['hidden_states']

            # Only take the embeddings from the second-to-last layer
            token_embeddings = hidden_states[-2]

            # Calculate the average embedding
            avg_embedding = token_embeddings.mean(dim=1)
            embeddings.append(avg_embedding.cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings


def extract_faus(of_bin_loc, vid_loc, out_dir):
    """Extract facial action units from video using OpenFace
    Example of_big_loc: /home/mason/OpenFace/bin/FaceLandmarkVidMulti"""

    command = f"{of_bin_loc} -f {vid_loc} -out_dir {out_dir} -aus"
    print(command)
    os.system(command)


def extract_acoustic_features(signal, base_fname):
    """
    Extract acoustic features from the audio signal for a specific speaker.

    Args:
    signal (numpy array): The audio signal for a specific speaker.
    window_size (int): Size of the rolling window.
    step_size (int): Step size for the rolling window.
    base_fname (str): Base filename for saving the features.
    speaker_label (str): Label for the speaker (e.g., 's1', 's2')
                         to be included in the filename.

    Returns:
    None
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    features = smile.process_signal(signal=signal, sampling_rate=16000)
    features.to_csv(os.path.join(acoustic_dir,
                                 f"{base_fname}.csv"))


def extract_linguistic_features(rolling_window, use_auth_token, output_fname):
    """
    Extract linguistic features from the audio rolling window.

    Args:
    rolling_window (numpy array): Rolling window of the speaker signal.
    use_auth_token (str): Authentication token for transcription service.
    output_fname (str): Filename for saving the extracted linguistic features.

    Returns:
    None
    """
    transcription_output = transcribe_rolling_window(rolling_window,
                                                     use_auth_token)
    transcription_output = [entry['text'] for entry in transcription_output]

    linguistic_features = gen_sentence_embeddings(transcription_output)
    np.save(os.path.join(linguistic_dir, f"{output_fname}.npy"),
            linguistic_features)


def extract_facial_features(of_bin_loc, video_dir, base_fname):
    """
    Extract facial features using OpenFace.

    Args:
    of_bin_loc (str): Path to the OpenFace binary.
    video_dir (str): Directory containing the video files.
    base_fname (str): Base filename for saving the features.

    Returns:
    None
    """
    vid_loc = os.path.join(video_dir, f"{base_fname}.mp4")
    extract_faus(of_bin_loc, vid_loc, facial_dir)

    facial_features_path = os.path.join(facial_dir, f"{base_fname}.csv")
    facial_features = pd.read_csv(facial_features_path)
    return facial_features
