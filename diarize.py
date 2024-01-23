import os
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from tqdm import tqdm
from utilities import load_wav_file


def diarize(file_path, use_auth_token, num_speakers=2, gpu=True):
    """
    Performs speaker diarization on a .wav file.

    Diarization segments the audio files into speaker turns and labels each
    segment with a speaker identifier.

    Args:
        directory (str): The directory containing .wav files to be processed.
        use_auth_token (str): Authentication token for accessing the
                              diarization model.
        num_speakers (int, optional): Expected number of speakers in the audio.
        gpu (bool, optional): Flag to use GPU for processing if available.

    Raises:
        AssertionError: If CUDA is not available when gpu is set to True.

    Returns:
        list: A list containing diarization information for each audio file.
              Each entry is a list of [start_time, end_time,
                                       speaker_label, file_name].
    """

    print("Diarizing audio")

    if gpu and not torch.cuda.is_available():
        raise AssertionError(
            "CUDA Not Available, \
                             set gpu=False in function or check that \
                                your torch install has CUDA enabled"
        )

    utterance_info = []  # Generate list to store output data

    # Import pretrained diarization pipeline from pyanote
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=use_auth_token
    )

    dia = pipeline(file_path, num_speakers=num_speakers)
    for speech_turn, _, speaker in dia.itertracks(yield_label=True):
        utterance_info.append([speech_turn.start, speech_turn.end, speaker, file_path])
    return utterance_info


def remove_overlap(diarization_info):
    """
    Processes diarization information to remove overlapping speech segments.

    Args:
    diarization_info (list): List of diarization data for each utterance.

    Returns:
    tuple: Two lists (NOR1, NOR2) representing non-overlapping ranges for each
           speaker.
    """
    ranges = {}

    for utterance in diarization_info:
        start = utterance[0]
        end = utterance[1]
        speaker = utterance[2]

        if speaker not in ranges.keys():
            ranges[speaker] = [(start, end)]
        else:
            ranges[speaker].append((start, end))

    speakers = list(ranges.keys())
    speakers.sort()
    print(f"Found speakers: {speakers}")

    if len(speakers) != 2:
        raise AssertionError("Expected only two speakers")

    # Generate lists of ranges
    R1 = ranges[speakers[0]]
    R2 = ranges[speakers[1]]

    # Initialize no overlap lists and itteration ranges
    NOR1 = []
    NOR2 = []
    max_i = len(R1)
    max_j = len(R2)
    i = 0
    j = 0

    a = R1[i][0]
    b = R1[i][1]
    c = R2[j][0]
    d = R2[j][1]

    # Define a helper function to break from loop
    def break_loop(i, j, max_i, max_j, NOR1=NOR1, NOR2=NOR2):
        bool_break = False
        if i == max_i:
            NOR2 = NOR2 + R2[j:max_j]
            bool_break = True

        if j == max_j:
            NOR1 = NOR1 + R1[i:max_i]
            bool_break = True
        return bool_break

    # Consider two ranges (a, b) and (c, d)
    while i < max_i and j < max_j:
        # No overlap case abcd
        if b <= c:
            NOR1.append((a, b))
            i += 1
            if break_loop(i, j, max_i, max_j):
                break
            a = R1[i][0]
            b = R1[i][1]
            continue

        # No overlap case cdab
        if d <= a:
            NOR2.append((c, d))
            j += 1
            if break_loop(i, j, max_i, max_j):
                break
            c = R2[j][0]
            d = R2[j][1]
            continue

        if a < c and c < b:
            # Overlap case acbd
            if b < d:
                NOR1.append((a, c))
                NOR2.append((b, d))
                i += 1
                j += 1
                if break_loop(i, j, max_i, max_j):
                    break
                a = R1[i][0]
                b = R1[i][1]
                c = R2[j][0]
                d = R2[j][1]
                continue

            # Overlap case acdb
            if d < b:
                NOR1.append((a, c))
                a = d
                j += 1
                if break_loop(i, j, max_i, max_j):
                    break
                c = R2[j][0]
                d = R2[j][1]
                continue

        if c < a and a < d:
            # Overlap case cadb
            if d < b:
                NOR1.append((d, b))
                NOR2.append((c, a))
                i += 1
                j += 1
                if break_loop(i, j, max_i, max_j):
                    break
                a = R1[i][0]
                b = R1[i][1]
                c = R2[j][0]
                d = R2[j][1]
                continue

            # Overlap case cabd
            if c < b and b < d:
                NOR2.append((c, a))
                c = b
                i += 1
                if break_loop(i, j, max_i, max_j):
                    break
                break_loop(i, j, max_i, max_j)
                c = R1[i][0]
                d = R1[i][1]
                continue

    def filter_list_of_tuples(lst):
        """Create a new list that will hold the tuples
        that satisfy the condition"""

        filtered_lst = []

        # Iterate over the input list of tuples
        for tup in lst:
            if tup[0] != tup[1]:
                # If not, append the tuple to the filtered list
                filtered_lst.append(tup)

        # Return the filtered list
        return filtered_lst

    NOR1 = filter_list_of_tuples(NOR1)
    NOR2 = filter_list_of_tuples(NOR2)

    return NOR1, NOR2


def keep_ranges(signal, ranges, seconds=True, sample_rate=16000):
    """
    Extracts specified ranges from a signal, padding the rest with zeros.

    Args:
    signal (numpy.ndarray): The original audio signal.
    ranges (list of tuples): Time ranges to keep in the signal.
    seconds (bool, optional): Whether the ranges are in seconds (True)
                              or samples (False). Defaults to True.
    sample_rate (int, optional): The sample rate of the audio signal.
                                 Required if seconds is True.

    Returns:
    numpy.ndarray: The modified signal with specified ranges kept and
                   others zero-padded.
    """
    # If seconds is True, convert to samples
    if seconds:
        ranges = [(int(r[0] * sample_rate), int(r[1] * sample_rate)) for r in ranges]

    new_signal = np.zeros(len(signal))
    for tup in ranges:
        new_signal[tup[0] : tup[1]] = signal[tup[0] : tup[1]]
    return new_signal


def load_and_diarize_audio(base_fname, audio_dir, use_auth_token):
    """
    Loads an audio file and performs diarization to determine speaker segments.

    Args:
    base_fname (str): The base filename of the audio file (without extension).
    audio_dir (str): The directory where the audio file is located.
    use_auth_token (str): The authentication token for accessing the
                          diarization model.

    Returns:
    tuple: The audio signal, and two lists representing time ranges for two
           speakers (R1, R2).
    """
    signal = load_wav_file(base_fname + ".wav", audio_dir)
    file_path = os.path.join(audio_dir, base_fname + ".wav")
    diarization_info = diarize(file_path, use_auth_token)
    R1, R2 = remove_overlap(diarization_info)
    return signal, R1, R2
