import logging
import os
import torch
import wenet
from typing import Any, Optional, Tuple
from enum import Enum
from pyannote.core import Annotation
from .diarization import (
    run_diarization as core_run_diarization, # Alias to avoid name clash if needed
    load_diarization_pipeline as core_load_diarization_pipeline,
    DiarizationError
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model name for automatic download via HuggingFace
REVERB_MODEL_NAME = "reverb_asr_v1" # Corresponds to Rev's model
# TODO: Allow specifying local model path via config later?
# LOCAL_MODEL_PATH = None # Example: "/path/to/local/reverb-model-dir"

class TranscriptionError(Exception):
    """Custom exception for transcription errors."""
    pass

class DiarizeMode(Enum):
    OFF = "off"
    AUTO = "auto"
    MANUAL = "manual"

# Placeholder for the loaded Reverb model/pipeline
# This might be initialized once at application startup later
reverb_pipeline: Any | None = None

def load_reverb_model() -> None:
    """Loads the Reverb ASR model and configures the pipeline.

    TODO: Handle model path resolution (allow local path override).
          Configure GPU usage if available.
    """
    global reverb_pipeline
    if reverb_pipeline is None:
        logging.info(f"Loading Reverb ASR model ('{REVERB_MODEL_NAME}')... This may take time.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        try:
            # load_model downloads from HuggingFace if name is provided,
            # or loads from local path if path is provided.
            # We assume 'reverb_asr_v1' is the correct HuggingFace identifier.
            # TODO: Check if model needs specific config/checkpoint paths or just name
            reverb_pipeline = wenet.load_model(
                REVERB_MODEL_NAME,
                # gpu=0 if device == "cuda" else -1 # wenet might use gpu arg?
                # Check wenet.load_model signature if needed
            )
            # TODO: Verify how device is set - might be post-load or via load_model arg
            # If load_model doesn't handle device, might need:
            # reverb_pipeline.to(device)

            logging.info(f"Reverb ASR model '{REVERB_MODEL_NAME}' loaded successfully onto {device}.")
        except Exception as e:
            logging.exception(f"Failed to load Reverb ASR model '{REVERB_MODEL_NAME}'.")
            # Reset pipeline to ensure it's not partially loaded
            reverb_pipeline = None
            raise TranscriptionError(f"Failed to load ASR model: {e}")

def transcribe_audio(wav_file_path: str, 
                     diarize_mode: DiarizeMode = DiarizeMode.OFF, 
                     num_speakers_manual: Optional[int] = None) -> Tuple[str, Optional[Annotation]]:
    """Transcribes the given WAV file using the Reverb ASR model.

    Optionally performs speaker diarization before transcription.

    Args:
        wav_file_path: Path to the input WAV file (mono, 16kHz recommended).
        diarize_mode: Mode for speaker diarization (OFF, AUTO, MANUAL).
        num_speakers_manual: Explicit number of speakers if diarize_mode is MANUAL.

    Returns:
        A tuple containing:
            - The transcribed text (str).
            - Diarization result (Annotation object, or None if disabled/failed).

    Raises:
        TranscriptionError: If transcription fails.
        DiarizationError: If diarization is requested but fails.
    """
    global reverb_pipeline
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"Input WAV file not found: {wav_file_path}")

    # Ensure the model is loaded (ideally done once at startup)
    load_reverb_model()

    if reverb_pipeline is None:
         raise TranscriptionError("Reverb ASR model is not loaded.")

    logging.info(f"Starting transcription for: {wav_file_path}")
    logging.info(f"Using speaker count setting: {'Auto' if diarize_mode == DiarizeMode.AUTO else 'Manual' if diarize_mode == DiarizeMode.MANUAL else 'Off'}")

    diarization_info: Optional[Annotation] = None
    diarization_error_msg: Optional[str] = None

    # --- Diarization Step (Optional) ---
    if diarize_mode != DiarizeMode.OFF:
        # --- Load Diarization Pipeline (only if needed) --- 
        try:
            core_load_diarization_pipeline() # Ensure it's loaded
        except DiarizationError as e:
            logging.warning(f"Could not load diarization pipeline: {e}")
            # Treat this as a failure for the diarization step
            diarization_error_msg = f"Diarization Failed: Could not load pipeline - {e}"
            diarization_info = None
            # Skip the rest of the diarization attempt if loading failed
            # Proceed to transcription, but with the diarization error message.
            # We'll check diarization_error_msg before trying to run diarization.

        logging.info(f"Diarization requested (Mode: {diarize_mode.name})")
        num_speakers_to_pass = num_speakers_manual if diarize_mode == DiarizeMode.MANUAL else None
        if diarize_mode == DiarizeMode.AUTO:
            logging.info("Auto-detecting speaker count.")
        elif num_speakers_to_pass:
             logging.info(f"Using manual speaker count: {num_speakers_to_pass}")

        # --- Run Diarization (only if loading succeeded) ---
        if diarization_error_msg is None: # Only run if loading didn't already fail
            try:
                # Call the *actual* run_diarization from the core module
                diarization_info = core_run_diarization(wav_file_path, num_speakers=num_speakers_to_pass)
                logging.info(f"core_run_diarization returned object of type: {type(diarization_info)}")

                if diarization_info:
                    logging.info("Diarization successful.")
                else:
                    # Handle case where run_diarization returns None or empty annotation
                    logging.warning("Diarization process returned no annotation.")
                    diarization_info = None
                    diarization_error_msg = "Diarization completed but produced no speaker segments."

            except DiarizationError as e:
                logging.error(f"Diarization failed during run: {e}")
                diarization_error_msg = f"Diarization Failed: {e}"
                diarization_info = None
            except Exception as e:
                logging.exception(f"An unexpected error occurred during diarization: {e}")
                diarization_error_msg = f"Diarization Failed: Unexpected error - {e}"
                diarization_info = None
        # else: diarization_error_msg already contains the loading failure message

    # --- Transcription Step ---
    # Parameters based on spec and assumptions from wenet usage
    # NOTE: Diarization params removed as it uses a separate pyannote pipeline.
    # These might need refinement based on actual wenet/reverb API for transcribe
    transcribe_params = {
        # --- Apply Hidden Defaults from Spec (Map to wenet params - Assumptions!) ---
        "beam_size": 12,        # Mapped from beam_width (Assumption)
        "ctc_weight": 1.5,      # Mapped from lm_weight (Assumption)
        # --- Other potential params? ---
        # "verbatimicity": 0.5, # Example from wenet docs, not in our spec
    }
    # Filter out None values as SDK might not like them
    transcribe_params = {k: v for k, v in transcribe_params.items() if v is not None}
    logging.debug(f"Transcription parameters: {transcribe_params}")

    try:
        # Call the transcribe method of the loaded wenet model
        # Pass the audio file path and parameters
        # Note: The exact parameter names (diarize, num_speakers, beam_size, ctc_weight)
        # are based on assumptions and might need correction.
        result = reverb_pipeline.transcribe(
            wav_file_path,
            **transcribe_params
        )

        # The transcribe function directly returns the transcript string
        if isinstance(result, str):
            transcript_text = result
            logging.debug("Transcription returned a string result.")
        else:
            # Log an error if the result is not a string as expected
            logging.error(
                f"Unexpected transcription result type: {type(result)}. Expected str."
            )
            raise TranscriptionError(
                f"Unexpected result format from transcription: {type(result)}"
            )

        logging.info(f"Transcription successful for: {wav_file_path}")
    except Exception as e:
        logging.exception(f"Transcription failed for {wav_file_path}: {e}")
        error_suffix = f" (Diarization also failed: {diarization_error_msg})" if diarization_error_msg else ""
        raise TranscriptionError(f"Transcription failed: {e}{error_suffix}") from e

    # Return transcript and the Annotation object (or None if failed/disabled)
    # If diarization failed, return the error message string instead of the Annotation object
    # The caller (GUI) will need to handle displaying the error or formatting the Annotation.
    return transcript_text, diarization_info if diarization_info else diarization_error_msg

def format_diarization_output(annotation: Annotation) -> str:
    """Formats the pyannote Annotation object into a human-readable string.

    Args:
        annotation: The Annotation object from pyannote.

    Returns:
        A string listing speaker segments chronologically.
    """
    if not annotation:
        return "No diarization data available."

    output_lines = []
    # pyannote's Annotation object gives segments per speaker.
    # We want to iterate through time, so we use timeline(with_labels=True)
    for segment, _, speaker_label in annotation.itertracks(yield_label=True):
        start_s = segment.start
        end_s = segment.end
        # Format time like [HH:MM:SS.fff - HH:MM:SS.fff] SPEAKER_ID
        start_hms = f"{int(start_s // 3600):02d}:{int((start_s % 3600) // 60):02d}:{start_s % 60:06.3f}"
        end_hms = f"{int(end_s // 3600):02d}:{int((end_s % 3600) // 60):02d}:{end_s % 60:06.3f}"
        output_lines.append(f"[{start_hms} - {end_hms}] {speaker_label}")

    # The segments might not be perfectly chronological if speakers interleave briefly,
    # although itertracks usually handles this. Sorting ensures order.
    # We sort based on the start time extracted from the formatted string.
    output_lines.sort(key=lambda line: float(line.split(':')[2].split(' ')[0]))

    return "\n".join(output_lines)
