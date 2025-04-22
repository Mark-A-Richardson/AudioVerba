import os
import subprocess
import logging
import tempfile

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FFMPEG_PATH = "ffmpeg" # Assuming ffmpeg is in PATH for now

class ConversionError(Exception):
    """Custom exception for FFmpeg conversion errors."""
    pass

def convert_to_wav(input_file_path: str) -> str:
    """Converts an audio/video file to 16kHz, 16-bit PCM mono WAV using FFmpeg.

    Args:
        input_file_path: Path to the input audio or video file.

    Returns:
        Path to the generated WAV file in a temporary directory.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ConversionError: If FFmpeg command fails or FFmpeg is not found.
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    output_dir = tempfile.gettempdir()
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_file_path = os.path.join(output_dir, f"{base_name}_converted.wav")

    command = [
        FFMPEG_PATH,
        "-i", input_file_path,
        "-ar", "16000",         # Audio rate: 16 kHz
        "-ac", "1",             # Audio channels: 1 (mono)
        "-c:a", "pcm_s16le",   # Codec: PCM signed 16-bit little-endian
        "-y",                  # Overwrite output file if it exists
        output_file_path
    ]

    logging.info(f"Running FFmpeg command: {' '.join(command)}")

    try:
        # Use capture_output=True to get stdout/stderr
        result = subprocess.run(
            command,
            check=True, # Raises CalledProcessError on non-zero exit code
            capture_output=True,
            text=True # Decode stdout/stderr as text
        )
        logging.info(f"FFmpeg conversion successful for {input_file_path}")
        logging.debug(f"FFmpeg stdout: {result.stdout}")
        logging.debug(f"FFmpeg stderr: {result.stderr}")
        return output_file_path
    except FileNotFoundError:
        logging.error(f"FFmpeg command '{FFMPEG_PATH}' not found. Is FFmpeg installed and in PATH?")
        raise ConversionError(f"FFmpeg not found at '{FFMPEG_PATH}'. Ensure it's installed and in PATH.")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion failed for {input_file_path}. Exit code: {e.returncode}")
        logging.error(f"FFmpeg stderr: {e.stderr}")
        raise ConversionError(f"FFmpeg failed: {e.stderr}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during conversion: {e}")
        raise ConversionError(f"An unexpected error occurred: {e}")
