"""
Speaker Diarization Module using pyannote.audio.

This module handles loading the diarization model and processing audio files
to identify speaker segments.
"""

import logging
import os
from typing import Optional
from functools import lru_cache

import torch
from huggingface_hub import hf_hub_download
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from .utils import get_device

# Constants
DIARIZATION_MODEL_REPO = "pyannote/speaker-diarization-3.1"
DIARIZATION_MODEL_ALIAS = "dia"
CONFIG_YAML_FILENAME = "config.yaml"

# Global variable to cache the pipeline
_diarization_pipeline: Optional[Pipeline] = None

class DiarizationError(Exception):
    """Custom exception for diarization errors."""
    pass

def _get_auth_token() -> Optional[str]:
    """Retrieves the Hugging Face auth token from environment variables."""
    return os.environ.get("HUGGING_FACE_HUB_TOKEN")

@lru_cache(maxsize=1) # Cache the pipeline after first load
def load_diarization_pipeline() -> Pipeline:
    """Loads the diarization pipeline, potentially using a cached version.

    Requires the HUGGING_FACE_HUB_TOKEN environment variable to be set for
    accessing the gated model.

    Returns:
        The loaded pyannote.audio Pipeline object.

    Raises:
        DiarizationError: If the model cannot be loaded or token is invalid/missing.
    """
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        logging.info("Using cached diarization pipeline.")
        return _diarization_pipeline

    # --- Explicitly get the token --- #
    hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not hf_token:
        logging.error("Hugging Face Hub token not found in environment variable HUGGING_FACE_HUB_TOKEN.")
        raise DiarizationError("Hugging Face authentication token not found. Please set HUGGING_FACE_HUB_TOKEN.")
    else:
        logging.info("Hugging Face Hub token found. Proceeding with pipeline loading.")

    try:
        # Get the appropriate device (cuda or cpu) using the utility function
        # The utility function handles logging the device choice.
        device: torch.device = get_device()

        # Download config.yaml first to potentially get class hints
        hf_hub_download(
            repo_id=DIARIZATION_MODEL_REPO,
            filename=CONFIG_YAML_FILENAME,
            use_auth_token=hf_token, # Pass the token explicitly
        )

        # Load the pipeline
        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL_REPO,
            use_auth_token=hf_token # Pass the token explicitly
        )
        # Move pipeline to the selected device
        pipeline.to(device)
        _diarization_pipeline = pipeline
        logging.info("Diarization pipeline loaded successfully.")
        return _diarization_pipeline
    except Exception as e:
        logging.exception(f"Failed to load diarization pipeline: {e}")
        raise DiarizationError(f"Could not load diarization pipeline: {e}")

def run_diarization(wav_path: str, num_speakers: Optional[int] = None) -> Annotation:
    """Performs speaker diarization on a WAV file.

    Args:
        wav_path: Path to the input WAV file (mono, 16kHz recommended).
        num_speakers: The number of speakers to detect. If None, the model
                      attempts to detect automatically.

    Returns:
        A pyannote.core.Annotation object containing speaker segments.

    Raises:
        DiarizationError: If the pipeline hasn't been loaded or processing fails.
    """
    global _diarization_pipeline
    if not _diarization_pipeline:
        # Attempt to load with default token mechanism if not pre-loaded
        load_diarization_pipeline()
        if not _diarization_pipeline:
             raise DiarizationError("Diarization pipeline not loaded. Call load_diarization_pipeline() first.")

    logging.info(f"Running diarization on: {wav_path}")
    if num_speakers:
        logging.info(f"Expecting {num_speakers} speakers.")
        params = {"num_speakers": num_speakers}
    else:
        logging.info("Attempting to automatically detect number of speakers.")
        # Default behavior of the pipeline if num_speakers is not provided
        # Or we can explicitly pass None if the pipeline requires it
        params = {}

    try:
        # The pipeline expects a dictionary or path-like object
        # Ensure the path is valid and accessible
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Input WAV file not found: {wav_path}")

        diarization_result: Annotation = _diarization_pipeline(wav_path, **params)
        logging.info(f"Diarization complete for: {wav_path}")
        return diarization_result
    except Exception as e:
        logging.exception(f"Error during diarization processing for {wav_path}: {e}")
        raise DiarizationError(f"Diarization processing failed: {e}")

def format_diarization_output(annotation: Annotation) -> str:
    """Formats the pyannote Annotation object into a simple string."""
    output_lines = []
    # Sort segments by start time
    for segment, track, label in annotation.itertracks(yield_label=True):
        start_s = segment.start
        end_s = segment.end
        # Format time as MM:SS.ms
        start_min, start_sec = divmod(start_s, 60)
        end_min, end_sec = divmod(end_s, 60)
        time_str = f"[{int(start_min):02d}:{start_sec:06.3f} - {int(end_min):02d}:{end_sec:06.3f}]"
        output_lines.append(f"{time_str} {label}")
    return "\n".join(output_lines)

# Example usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # This requires a valid HUGGING_FACE_HUB_TOKEN environment variable
    # and a sample audio file (e.g., 'sample.wav')
    TEST_WAV = 'path/to/your/test_audio.wav' # CHANGE THIS
    EXPECTED_SPEAKERS = None # Or set to an integer e.g. 2

    if not os.path.exists(TEST_WAV):
        print(f"Test audio file not found: {TEST_WAV}")
        print("Please update the TEST_WAV variable in diarization.py")
    else:
        try:
            print("Loading pipeline...")
            load_diarization_pipeline()
            print("Running diarization...")
            result = run_diarization(TEST_WAV, num_speakers=EXPECTED_SPEAKERS)
            print("\nDiarization Result:")
            print(format_diarization_output(result))

            # Print speaker turns summary
            # print("\nSpeaker Turns Summary:")
            # print(result.chart())

        except DiarizationError as e:
            print(f"Diarization failed: {e}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
