import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from pyannote.core import Annotation, Segment

# Modules and functions to test
from audioverba.core.transcription import (
    transcribe_audio, 
    format_diarization_output, 
    DiarizeMode, 
    TranscriptionError,
)
from audioverba.core.diarization import DiarizationError

# --- Fixtures --- 

@pytest.fixture
def mock_annotation() -> Annotation:
    """Create a dummy pyannote Annotation object for testing formatting."""
    annotation = Annotation()
    annotation[Segment(0.5, 1.5)] = 'SPEAKER_00'
    annotation[Segment(1.8, 2.9)] = 'SPEAKER_01'
    annotation[Segment(3.0, 4.2)] = 'SPEAKER_00'
    return annotation

@pytest.fixture
def dummy_wav_path(tmp_path: Path) -> str:
    """Creates a dummy empty WAV file path for transcription tests."""
    p = tmp_path / "test.wav"
    p.touch()
    return str(p)

# --- Test Cases --- 

# TODO: Add tests for transcribe_audio with mocks

# Test format_diarization_output
def test_format_diarization_output_with_data(mock_annotation: Annotation):
    """Test formatting a valid Annotation object."""
    expected_output = (
        "[00:00:00.500 - 00:00:01.500] SPEAKER_00\n"
        "[00:00:01.800 - 00:00:02.900] SPEAKER_01\n"
        "[00:00:03.000 - 00:00:04.200] SPEAKER_00"
    )
    # Note: Original implementation sorted, so let's ensure fixture order matches expected.
    # If the implementation relied on sorting, the fixture order wouldn't matter.
    formatted = format_diarization_output(mock_annotation)
    assert formatted == expected_output

def test_format_diarization_output_empty():
    """Test formatting an empty Annotation object."""
    empty_annotation = Annotation()
    formatted = format_diarization_output(empty_annotation)
    # Depending on implementation, might return empty string or a specific message
    # Let's assume it returns a specific message based on current code review
    assert formatted == "No diarization data available." # Or check if it's just empty

def test_format_diarization_output_none():
    """Test formatting when None is passed (should be handled gracefully)."""
    # This case might not be strictly necessary if type hinting prevents None,
    # but good for robustness if None could sneak in.
    # Based on current implementation, it might raise an error or return the message.
    # Let's assume it returns the message. Adjust if needed.
    assert format_diarization_output(None) == "No diarization data available."


# --- Tests for transcribe_audio (using mocks) ---

@patch('audioverba.core.transcription.core_load_diarization_pipeline')
@patch('audioverba.core.transcription.core_run_diarization')
@patch('audioverba.core.transcription.reverb_pipeline')
@patch('audioverba.core.transcription.load_reverb_model') # Mock model loading
def test_transcribe_audio_diarization_off(
    mock_load_reverb: MagicMock,
    mock_reverb_pipeline: MagicMock,
    mock_run_diarization: MagicMock,
    mock_load_diarization: MagicMock,
    dummy_wav_path: str
):
    """Test transcribe_audio with diarization mode OFF."""
    mock_reverb_pipeline.transcribe.return_value = "This is a transcript."
    
    transcript, diarization_info = transcribe_audio(dummy_wav_path, diarize_mode=DiarizeMode.OFF)

    mock_load_reverb.assert_called_once() # Ensure ASR model load was attempted
    mock_reverb_pipeline.transcribe.assert_called_once_with(
        dummy_wav_path, beam_size=12, ctc_weight=1.5
    )
    mock_load_diarization.assert_not_called() # Diarization load should NOT be called
    mock_run_diarization.assert_not_called()  # Diarization run should NOT be called

    assert transcript == "This is a transcript."
    assert diarization_info is None # Expect None when OFF


@patch('audioverba.core.transcription.core_load_diarization_pipeline')
@patch('audioverba.core.transcription.core_run_diarization')
@patch('audioverba.core.transcription.reverb_pipeline')
@patch('audioverba.core.transcription.load_reverb_model') 
def test_transcribe_audio_diarization_auto_success(
    mock_load_reverb: MagicMock,
    mock_reverb_pipeline: MagicMock,
    mock_run_diarization: MagicMock,
    mock_load_diarization: MagicMock,
    dummy_wav_path: str,
    mock_annotation: Annotation
):
    """Test transcribe_audio with diarization mode AUTO (success)."""
    mock_reverb_pipeline.transcribe.return_value = "Transcript text."
    mock_run_diarization.return_value = mock_annotation # Simulate successful diarization
    
    transcript, diarization_info = transcribe_audio(dummy_wav_path, diarize_mode=DiarizeMode.AUTO)

    mock_load_reverb.assert_called_once()
    mock_reverb_pipeline.transcribe.assert_called_once_with(
        dummy_wav_path, beam_size=12, ctc_weight=1.5
    )
    mock_load_diarization.assert_called_once() # Diarization load SHOULD be called
    mock_run_diarization.assert_called_once_with(dummy_wav_path, num_speakers=None) # num_speakers=None for AUTO

    assert transcript == "Transcript text."
    assert diarization_info == mock_annotation


@patch('audioverba.core.transcription.core_load_diarization_pipeline')
@patch('audioverba.core.transcription.core_run_diarization')
@patch('audioverba.core.transcription.reverb_pipeline')
@patch('audioverba.core.transcription.load_reverb_model') 
def test_transcribe_audio_diarization_manual_success(
    mock_load_reverb: MagicMock,
    mock_reverb_pipeline: MagicMock,
    mock_run_diarization: MagicMock,
    mock_load_diarization: MagicMock,
    dummy_wav_path: str,
    mock_annotation: Annotation
):
    """Test transcribe_audio with diarization mode MANUAL (success)."""
    mock_reverb_pipeline.transcribe.return_value = "Manual transcript."
    mock_run_diarization.return_value = mock_annotation
    num_speakers_manual = 2
    
    transcript, diarization_info = transcribe_audio(
        dummy_wav_path, 
        diarize_mode=DiarizeMode.MANUAL, 
        num_speakers_manual=num_speakers_manual
    )

    mock_load_reverb.assert_called_once()
    mock_reverb_pipeline.transcribe.assert_called_once_with(
        dummy_wav_path, beam_size=12, ctc_weight=1.5
    )
    mock_load_diarization.assert_called_once()
    mock_run_diarization.assert_called_once_with(dummy_wav_path, num_speakers=num_speakers_manual)

    assert transcript == "Manual transcript."
    assert diarization_info == mock_annotation


@patch('audioverba.core.transcription.core_load_diarization_pipeline')
@patch('audioverba.core.transcription.core_run_diarization', side_effect=DiarizationError("Diarization failed"))
@patch('audioverba.core.transcription.reverb_pipeline')
@patch('audioverba.core.transcription.load_reverb_model') 
def test_transcribe_audio_diarization_auto_failure(
    mock_load_reverb: MagicMock,
    mock_reverb_pipeline: MagicMock,
    mock_run_diarization: MagicMock,
    mock_load_diarization: MagicMock,
    dummy_wav_path: str
):
    """Test transcribe_audio with diarization mode AUTO where diarization fails."""
    mock_reverb_pipeline.transcribe.return_value = "Transcript despite diarization fail."
    
    # The function should catch DiarizationError and return the error message string
    transcript, diarization_msg = transcribe_audio(dummy_wav_path, diarize_mode=DiarizeMode.AUTO)

    mock_load_reverb.assert_called_once()
    mock_reverb_pipeline.transcribe.assert_called_once_with(
        dummy_wav_path, beam_size=12, ctc_weight=1.5
    )
    mock_load_diarization.assert_called_once()
    mock_run_diarization.assert_called_once_with(dummy_wav_path, num_speakers=None)

    assert transcript == "Transcript despite diarization fail."
    assert isinstance(diarization_msg, str) # Expecting the error message string
    assert "Diarization failed" in diarization_msg


@patch('audioverba.core.transcription.reverb_pipeline') 
@patch('audioverba.core.transcription.load_reverb_model')
def test_transcribe_audio_transcription_failure(
    mock_load_reverb: MagicMock,
    mock_reverb_pipeline: MagicMock,
    dummy_wav_path: str
):
    """Test transcribe_audio when the core transcription call fails."""
    mock_reverb_pipeline.transcribe.side_effect = Exception("ASR engine exploded")
    
    with pytest.raises(TranscriptionError, match="ASR engine exploded"):
        transcribe_audio(dummy_wav_path, diarize_mode=DiarizeMode.OFF)
    
    mock_load_reverb.assert_called_once()
    mock_reverb_pipeline.transcribe.assert_called_once_with(
        dummy_wav_path, beam_size=12, ctc_weight=1.5
    )
 
 
 # Add more tests? e.g., transcription success but diarization load fails?
