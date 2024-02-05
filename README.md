# Multimodal Feature Extraction Pipeline

This pipeline processes video files to perform speaker diarization and extract combined multimodal features (acoustic, linguistic, and facial features) using tools like pyannote.audio, OpenSmile, and OpenFace.

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip (Python package manager)
- [FFmpeg](https://ffmpeg.org/download.html) (for video processing)

### Setting up the Environment

1. Clone the repository:
   ```bash
   git clone git@github.com:masonhargrave/MultimodalFeatureSync.git
   cd multimodal-feature-extraction
   ```

2. Create a virtual environment and activate it:
   ```bash
   conda create --name mfs python==3.8.15
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Additional Dependencies

Some components of this pipeline require special installation procedures. Follow the links below for installation instructions:

- [pyannote.audio Installation](https://github.com/pyannote/pyannote-audio)
- [OpenSmile Installation](https://github.com/audeering/opensmile-python)
- [OpenFace Installation](https://github.com/TadasBaltrusaitis/OpenFace)

## Configuration

### Environment Variables

Set the following environment variables with your specific configuration:

- `PYANNOTE_AUTH_TOKEN`: Your authentication token for pyannote.audio.
- `OPENFACE_BIN_LOC`: The path to your OpenFace binary.

On Linux or macOS, you can set these variables in your shell:

```bash
export PYANNOTE_AUTH_TOKEN="your_token_here"
export OPENFACE_BIN_LOC="/path/to/OpenFace/bin/FeatureExtraction"
```

On Windows, use:

```cmd
set PYANNOTE_AUTH_TOKEN=your_token_here
set OPENFACE_BIN_LOC=\path\to\OpenFace\bin\FeatureExtraction
```

To accurately reflect the changes in your script, the usage instructions in your README file need to be updated. The new instructions should specify that the diarization stage does not require the CSV file, while the feature extraction stage does. Here's the updated usage section:

## Usage

To use the pipeline, run `process_pipeline.py` with the appropriate arguments:

- For diarization:
  Process all video files in a specified directory for diarization. No need for the CSV file in this stage.
  ```bash
  python process_pipeline.py /path/to/mp4/files/ --stage diarization
  ```

- For feature extraction (after diarization):
  Process all video files in a specified directory for feature extraction. This stage requires the path to a CSV file containing speaker labels to identify the patient's speech.
  ```bash
  python process_pipeline.py /path/to/mp4/files/ /path/to/speaker_labels.csv --stage features
  ```

Ensure to replace `/path/to/mp4/files/` with the actual path to your directory containing MP4 files and `/path/to/speaker_labels.csv` with the path to your CSV file containing the speaker labels. We have included `example_speaker_labels` so you can create your own.

## License

Copyright 2023 Mason Hargrave

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
