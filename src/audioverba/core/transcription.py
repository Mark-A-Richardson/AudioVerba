import logging
from wenet.cli.reverb import ReverbASR
from typing import Optional
from enum import Enum
from pyannote.core import Annotation
from .diarization import (
    run_diarization as core_run_diarization,
    load_diarization_pipeline as core_load_diarization_pipeline
)
from .exceptions import TranscriptionError, DiarizationError
from .utils import get_device
from pathlib import Path
from huggingface_hub import hf_hub_download
from PySide6.QtCore import QObject, Signal

# Setup basic logging
# Consider moving configuration to main.py or a dedicated logging setup module
logger = logging.getLogger(__name__) # Use module-specific logger

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "reverb_asr_v1"
CONFIG_PATH = MODEL_DIR / "config.yaml"
CHECKPOINT_PATH = MODEL_DIR / "reverb_asr_v1.pt"

# --- Hugging Face Download Constants ---
HF_REPO_ID = "Revai/reverb-asr" # Use user-provided repo ID
MODEL_FILENAMES = ["config.yaml", "reverb_asr_v1.pt", "tk.units.txt", "en-cmvn.json"] # Added tokenizer file and CMVN file
# ---------------------------------------

class DiarizeMode(Enum):
    """Modes for controlling speaker diarization."""
    OFF = "off"
    AUTO = "auto"
    MANUAL = "manual"

# Placeholder for the loaded Reverb model/pipeline
reverb_pipeline: ReverbASR | None = None

# --- Internal Signal Emitter --- #
class TranscriptionSignals(QObject):
    """Provides signals for transcribe_audio results."""
    transcript_ready = Signal(str) # Emits the final transcript text
    diarization_ready = Signal(object) # Emits str (formatted or error) or None

# --- Helper Function for Timestamp Formatting --- #

def _format_seconds(seconds: float) -> str:
    """Converts seconds into HH:MM:SS.ms format."""
    if seconds < 0:
        seconds = 0
    # Round before converting to int to handle floating point precision near millisecond boundaries
    milliseconds = int(round((seconds - int(seconds)) * 1000))

    # Handle potential rounding overflow (e.g., 0.9995s rounding to 1000ms)
    if milliseconds == 1000:
        milliseconds = 0
        seconds += 1 # Increment the whole second part

    seconds_int = int(seconds)
    minutes, seconds_rem = divmod(seconds_int, 60)
    hours, minutes_rem = divmod(minutes, 60)
    return f"{hours:02d}:{minutes_rem:02d}:{seconds_rem:02d}.{milliseconds:03d}"

# --- New Download Function ---
def _download_reverb_model_files() -> None:
    """Downloads Reverb ASR model files from Hugging Face Hub if they don't exist locally."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure target directory exists

    for filename in MODEL_FILENAMES:
        local_path = MODEL_DIR / filename
        if not local_path.is_file():
            logger.info(f"Downloading {filename} from {HF_REPO_ID} to {local_path}...")
            try:
                # Using local_dir ensures the file lands directly in our MODEL_DIR
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False, # Use file copy instead of symlinks
                    # cache_dir=None, # Optionally prevent using HF cache
                    force_download=False # Don't redownload if somehow already cached
                )
                # Verify download explicitly
                if not local_path.is_file():
                     raise FileNotFoundError(f"Download reported success but file not found at {local_path}")
                logger.info(f"Successfully downloaded {filename}.")
            except Exception as e:
                logger.error(f"Failed to download {filename} from {HF_REPO_ID}: {e}", exc_info=True)
                # Raise an error to prevent proceeding without the file
                raise FileNotFoundError(f"Failed to download required model file: {filename}") from e
        else:
            logger.debug(f"Model file {filename} already exists locally at {local_path}.")
# ---------------------------

def load_reverb_model() -> None:
    """
    Loads the Reverb ASR model directly using ReverbASR, enabling GPU.
    Downloads model files if they are not present locally.
    """
    global reverb_pipeline
    if reverb_pipeline is None:
        # --- Call Download Function ---
        try:
            logger.info("Checking for local Reverb ASR model files...")
            _download_reverb_model_files()
            logger.info("Model files are present locally.")
        except FileNotFoundError as e:
            # Log the specific download error and wrap it
            logger.error(f"Cannot load Reverb model due to download failure: {e}")
            raise TranscriptionError(f"A required model file could not be downloaded: {e}") from e
        except Exception as e:
            # Catch unexpected errors during download/check
            logger.error(f"An unexpected error occurred while preparing model files: {e}", exc_info=True)
            raise TranscriptionError(f"Failed to prepare model files: {e}") from e
        # ---------------------------

        # Proceed with loading now that files should exist
        # (File existence checks are now slightly redundant but act as a safeguard)
        if not CONFIG_PATH.is_file():
            msg = f"Reverb config file missing after check/download attempt: {CONFIG_PATH}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if not CHECKPOINT_PATH.is_file():
            msg = f"Reverb checkpoint file missing after check/download attempt: {CHECKPOINT_PATH}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Determine device and GPU ID
        device = get_device()
        gpu_id = 0 if device.type == 'cuda' else -1
        device_info = f"GPU {gpu_id}" if gpu_id != -1 else "CPU"
        logger.info(f"Attempting to load Reverb ASR model directly on {device_info} using local files...")

        try:
            # Instantiate ReverbASR directly
            reverb_pipeline = ReverbASR(
                str(CONFIG_PATH),           # Positional arg 1
                str(CHECKPOINT_PATH),       # Positional arg 2
                gpu=gpu_id,
                # Add other necessary args like overwrite_cmvn=False if needed
            )
            logger.info(f"Reverb ASR model loaded successfully on {device_info}.")

        except Exception as e:
            logger.error(f"Failed to load Reverb ASR model from local files: {e}", exc_info=True)
            reverb_pipeline = None # Ensure it's None if loading fails
            raise TranscriptionError(f"Failed to initialize Reverb ASR model: {e}") from e

# --- transcribe_audio function remains the same ---
def transcribe_audio(
    wav_file_path: str,
    signal_emitter: TranscriptionSignals,
    diarize_mode: DiarizeMode = DiarizeMode.OFF,
    num_speakers_manual: Optional[int] = None
) -> None:
    """
    Transcribes the given WAV file using the Reverb ASR model.

    Emits results via the provided signal_emitter.

    Args:
        wav_file_path: Path to the input WAV file (mono, 16kHz recommended).
        signal_emitter: QObject instance to emit results through.
        diarize_mode: Mode for speaker diarization (OFF, AUTO, MANUAL).
        num_speakers_manual: Explicit number of speakers if diarize_mode is MANUAL.

    # No return value

    Raises:
        # Errors should ideally be emitted via a signal too, or handled internally
        # For now, keep existing raises for critical failures like model loading
        TranscriptionError: If transcription model loading fails.
        DiarizationError: If diarization pipeline loading fails.
    """
    global reverb_pipeline
    diarization_info: Optional[Annotation] = None
    diarization_error_msg: Optional[str] = None
    transcript_text: str = ""

    # --- Ensure Models are Loaded ---
    # Load Reverb ASR model (handles download if needed)
    try:
        if reverb_pipeline is None:
            load_reverb_model()
        if reverb_pipeline is None: # Check again after loading attempt
             raise TranscriptionError("ASR model could not be loaded.")
    except (TranscriptionError, FileNotFoundError) as e:
        logger.error(f"Failed to load ASR model: {e}")
        raise TranscriptionError(f"Failed to load ASR model: {e}") from e # Re-raise consistently

    # Load Diarization pipeline only if needed
    diarization_pipeline = None
    if diarize_mode != DiarizeMode.OFF:
        try:
            diarization_pipeline = core_load_diarization_pipeline()
            if diarization_pipeline is None:
                raise DiarizationError("Diarization pipeline could not be loaded.")
        except (DiarizationError, Exception) as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            # Store error message, but allow transcription to proceed if possible
            diarization_error_msg = f"Diarization Skipped (Pipeline Load Failed): {e}"
            diarize_mode = DiarizeMode.OFF # Force disable if pipeline failed

    # --- Diarization Step (Optional) ---
    if diarize_mode != DiarizeMode.OFF and diarization_pipeline:
        try:
            logger.info(f"Running diarization (Manual Speakers: {num_speakers_manual})...")
            diarization_result = core_run_diarization(
                 wav_file_path,
                 num_speakers=num_speakers_manual
             )

            if isinstance(diarization_result, Annotation):
                diarization_info = diarization_result
                logger.info("Diarization successful.")
            elif isinstance(diarization_result, str): # Handle error string from run_diarization
                diarization_error_msg = f"Diarization Failed: {diarization_result}"
                diarization_info = None
                logger.error(diarization_error_msg)
            else:
                logger.warning("Diarization process returned unexpected type or no annotation.")
                diarization_info = None
                diarization_error_msg = "Diarization completed but produced no speaker segments."

        except DiarizationError as e:
            logger.error(f"Diarization failed during run: {e}")
            diarization_error_msg = f"Diarization Failed: {e}"
            diarization_info = None
        except Exception as e:
            logger.exception(f"An unexpected error occurred during diarization: {e}")
            diarization_error_msg = f"Diarization Failed: Unexpected error - {e}"
            diarization_info = None
    elif diarize_mode != DiarizeMode.OFF and not diarization_pipeline:
         # This case occurs if loading failed earlier and was recorded in diarization_error_msg
         logger.warning("Diarization skipped because the pipeline failed to load.")


    # --- Transcription Step ---
    # Parameters based on spec and assumptions from wenet usage
    # Note: These parameters might need refinement based on ReverbASR's actual transcribe method signature
    # Use inspect.signature(reverb_pipeline.transcribe) if unsure
    transcribe_params = {
        "beam_size": 12,        # From previous assumptions
        "ctc_weight": 1.5,      # From previous assumptions
        # Add other relevant parameters from the signature if needed:
        # "mode": 'ctc_prefix_beam_search', # Default from inspection
        # "verbatimicity": 1.0,            # Default from inspection
        # ... etc ...
    }
    # Filter out None values if necessary (though defaults usually handle this)
    # transcribe_params = {k: v for k, v in transcribe_params.items() if v is not None}
    logger.debug(f"Transcription parameters: {transcribe_params}")

    try:
        logger.info(f"Starting transcription for: {wav_file_path}")
        # Ensure pipeline is loaded before calling transcribe
        if reverb_pipeline is None:
            raise TranscriptionError("Cannot transcribe: ASR model is not loaded.")

        # Call the transcribe method of the loaded ReverbASR instance
        result = reverb_pipeline.transcribe(
            wav_file_path,
            **transcribe_params
        )

        if isinstance(result, str):
            transcript_text = result
            logger.info(f"Transcription successful for: {wav_file_path}")
            logger.debug(f"Transcript: {transcript_text[:100]}...") # Log snippet
        else:
            logger.error(f"Unexpected transcription result type: {type(result)}. Expected str.")
            raise TranscriptionError(f"Unexpected result format from transcription: {type(result)}")

    except Exception as e:
        logger.exception(f"Transcription failed for {wav_file_path}: {e}")
        error_suffix = f" (Diarization info: {diarization_error_msg or 'Success/Off'})"
        raise TranscriptionError(f"Transcription failed: {e}{error_suffix}") from e

    # --- Diarization Processing --- #
    # --- Calculate final diarization string --- #
    diarization_result_str: Optional[str] = None
    diarization_enabled = diarize_mode != DiarizeMode.OFF # Recalculate based on final mode

    if diarization_enabled:
        if diarization_info: # If Annotation object exists from earlier step
            logger.debug("Formatting successful diarization info.")
            diarization_result_str = format_diarization_output(diarization_info)
        elif diarization_error_msg: # If an error occurred during diarization steps
            logger.debug("Using diarization error message for result string.")
            # Ensure error message is clearly marked
            diarization_result_str = f"\n\n--------------------\n\nDiarization Info:\n{diarization_error_msg}"
        else: # Diarization was enabled but resulted in neither Annotation nor error message? Log warning.
            logger.warning("Diarization was enabled, but no Annotation object or error message was found.")
            diarization_result_str = "\n\n--------------------\n\nDiarization Info:\nNo segments found or unknown state."

    # --- Debug Log before emitting --- #
    logger.debug(
        f"Emitting from transcribe_audio. Final Diarization Result Type: {type(diarization_result_str).__name__}"
    )

    # Emit BASE transcript and the calculated diarization STRING/None
    signal_emitter.transcript_ready.emit(transcript_text)
    signal_emitter.diarization_ready.emit(diarization_result_str) # Emit None if diarization was off or failed cleanly


# --- Diarization Output Formatting --- #

def format_diarization_output(annotation: Annotation) -> str:
    """Formats the diarization Annotation object into a readable string (HH:MM:SS.ms)."""
    logger.debug(f"Formatting diarization for annotation object: {type(annotation)}")
    lines = [] # Remove header
    try:
        segment_count = 0
        # Sort segments by start time before formatting
        sorted_segments = sorted(annotation.itertracks(yield_label=True), key=lambda x: x[0].start)

        for segment, _, label in sorted_segments:
            segment_count += 1
            start_time = segment.start
            end_time = segment.end
            logger.debug(f"  Processing segment: {start_time:.3f}-{end_time:.3f} -> {label}")
            # Use the helper function for formatting
            formatted_start = _format_seconds(start_time)
            formatted_end = _format_seconds(end_time)
            lines.append(f"[{formatted_start} - {formatted_end}] {label}")

        if segment_count == 0:
            logger.warning("Annotation object yielded no segments.")
            # Return a specific message if no segments found, matching potential test cases
            return "No speaker segments found."

        logger.debug(f"Successfully formatted {segment_count} segments.")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error formatting diarization output: {e}", exc_info=True)
        # Consistent error message for easier testing
        return "Error during diarization formatting."