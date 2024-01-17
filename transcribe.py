import torch
import whisper
import numpy as np
from tqdm import tqdm


def transcribe_rolling_window(rolling_window,
                              return_full_output=False,
                              version="base"):
    """
    Transcribes audio segments using the Whisper model.

    This function processes each segment in a rolling window of audio data,
    transcribing the content using the Whisper machine learning model. It can
    return either the full output of the transcription, including additional
    metadata, or just the transcribed text.

    Args:
    rolling_window (numpy.ndarray): A 2D array where each row is an audio
                                    segment to be transcribed.
    return_full_output (bool, optional): If True, the function returns the
                                         full output of the Whisper model,
                                         including metadata. If False, only
                                         the transcribed text is returned.
                                         Defaults to False.
    version (str, optional): The version of the Whisper model to use.
                             Defaults to 'base'.

    Returns:
    list: If return_full_output is False, a list of strings, each representing
          the transcription of an audio segment.
          If return_full_output is True, a list of dictionaries containing
          full transcription outputs from Whisper.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    model = whisper.load_model(version).to(device)

    # Convert s1_rollowing_window to float32
    rolling_window = rolling_window.astype(np.float32)

    # Load s1_rolling_window into a torch tensor and send it to the GPU
    rolling_window = torch.from_numpy(rolling_window).cuda()

    # Transcribe each row of the rolling window

    full_output = []
    transcription_output = []

    for i in tqdm(range(rolling_window.shape[0])):
        transcription = model.transcribe(rolling_window[i])
        full_output.append(transcription)
        transcription_output.append(transcription["text"])

    if return_full_output:
        return full_output
    else:
        return transcription_output
