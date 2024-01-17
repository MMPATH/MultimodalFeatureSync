import torch
import whisper
import numpy as np
from tqdm import tqdm


def transcribe_rolling_window(rolling_window,
                              return_full_output=False,
                              version="base"):
    """Transcribe a rolling window of audio using Whisper"""

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
