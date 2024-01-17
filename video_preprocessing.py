import cv2
import os
import subprocess


def crop_video(source_path, temp_path):
    """
    Crops the left half of each frame in the given video file and saves it as
    a temporary file.

    Args:
    source_path (str): Path to the source video file to be cropped.
    temp_path (str): Path where the cropped temporary video file will be saved.

    Returns:
    bool: True if the video was successfully cropped and saved, False
          otherwise.
    """
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Error opening file {source_path}")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(temp_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[:, :frame_width]
        out.write(cropped_frame)

    cap.release()
    out.release()
    return True


def reencode_video(temp_path, target_path):
    """
    Re-encodes a video file using FFMPEG, typically used to process a
    temporary video file.

    Args:
    temp_path (str): Path to the temporary video file to be re-encoded.
    target_path (str): Path where the re-encoded video file will be saved.

    Returns:
    bool: True if the video was successfully re-encoded and saved,
          False otherwise.

    Note:
    This function uses the 'ffmpeg' command-line tool for video processing.
    The '-y' flag is used to overwrite the output file if it already exists.
    """
    command = [
        "ffmpeg",
        "-i",
        temp_path,
        "-c:v",
        "mpeg4",
        "-y",  # overwrite output file if it exists
        target_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"FFMPEG Error Output:\n{result.stderr.decode()}")
    return result.returncode == 0


def process_video(source_path, target_dir):
    """
    Process a single video file - cropping and re-encoding.

    Args:
    source_path (str): Path to the source video file.
    target_dir (str): Directory where the processed file will be saved.

    Returns:
    bool: True if processing is successful, False otherwise.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filename = os.path.basename(source_path)
    temp_path = os.path.join(target_dir, f"temp_{filename}")
    target_path = os.path.join(target_dir, filename)

    if crop_video(source_path, temp_path) and reencode_video(temp_path,
                                                             target_path):
        os.remove(
            temp_path
        )  # Remove the temporary file only if re-encoding is successful
        return True
    else:
        print(f"Re-encoding failed for {filename}")
        return False


def process_videos_in_directory(source_dir, target_dir):
    """
    Process all video files in a directory.

    Args:
    source_dir (str): Directory containing the video files to process.
    target_dir (str): Directory where the processed files will be saved.

    Returns:
    None
    """
    for filename in os.listdir(source_dir):
        if filename.endswith(".mp4"):
            source_path = os.path.join(source_dir, filename)
            process_video(source_path, target_dir)
