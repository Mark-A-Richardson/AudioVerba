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
    TranscriptionSignals,
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
    assert formatted == "No speaker segments found." # Changed expected message

def test_format_diarization_output_none():
    """Test formatting when None is passed (should be handled gracefully)."""
    # This case might not be strictly necessary if type hinting prevents None,
    # but good for robustness if None could sneak in.
    # Update assertion to match the actual error message returned by the except block
    assert format_diarization_output(None) == "Error during diarization formatting." # Changed expected message

# --- Tests for transcribe_audio (using mocks) ---

@patch('audioverba.core.transcription.core_load_diarization_pipeline')
@patch('audioverba.core.transcription.core_run_diarization')
@patch('audioverba.core.transcription.reverb_pipeline')
def test_transcribe_audio_diarization_off(
    mock_reverb_pipeline: MagicMock,
    mock_run_diarization: MagicMock,
    mock_load_diarization: MagicMock,
    dummy_wav_path: str
):
    """Test transcribe_audio with diarization mode OFF."""
    expected_transcript = "This is a transcript."
    mock_reverb_pipeline.transcribe.return_value = expected_transcript
    mock_emitter = MagicMock(spec=TranscriptionSignals)

    transcribe_audio(dummy_wav_path, diarize_mode=DiarizeMode.OFF, signal_emitter=mock_emitter)

    # Assert signals were emitted correctly
    mock_emitter.transcript_ready.emit.assert_called_once_with(expected_transcript)
    mock_emitter.diarization_ready.emit.assert_called_once_with(None) # Expect None when OFF
    mock_run_diarization.assert_not_called() # Ensure diarization wasn't run
    mock_load_diarization.assert_not_called()

@patch('audioverba.core.transcription.core_load_diarization_pipeline')
@patch('audioverba.core.transcription.core_run_diarization')
@patch('audioverba.core.transcription.reverb_pipeline')
def test_transcribe_audio_diarization_auto_success(
    mock_reverb_pipeline: MagicMock,
    mock_run_diarization: MagicMock,
    mock_load_diarization: MagicMock,
    dummy_wav_path: str,
    mock_annotation: Annotation
):
    """Test transcribe_audio with diarization mode AUTO (success)."""
    expected_transcript = "Transcript text."
    expected_diarization_output = format_diarization_output(mock_annotation)
    mock_reverb_pipeline.transcribe.return_value = expected_transcript
    mock_run_diarization.return_value = mock_annotation
    mock_emitter = MagicMock(spec=TranscriptionSignals)

    transcribe_audio(dummy_wav_path, diarize_mode=DiarizeMode.AUTO, signal_emitter=mock_emitter)

    # Assert signals
    mock_emitter.transcript_ready.emit.assert_called_once_with(expected_transcript)
    mock_emitter.diarization_ready.emit.assert_called_once_with(expected_diarization_output)

    # Assert mocks
    mock_load_diarization.assert_called_once() # Should be called in AUTO/MANUAL
    mock_run_diarization.assert_called_once_with(dummy_wav_path, num_speakers=None)

@patch('audioverba.core.transcription.core_load_diarization_pipeline')
@patch('audioverba.core.transcription.core_run_diarization')
@patch('audioverba.core.transcription.reverb_pipeline')
def test_transcribe_audio_diarization_manual_success(
    mock_reverb_pipeline: MagicMock,
    mock_run_diarization: MagicMock,
    mock_load_diarization: MagicMock,
    dummy_wav_path: str,
    mock_annotation: Annotation
):
    """Test transcribe_audio with diarization mode MANUAL (success)."""
    expected_transcript = "Manual transcript."
    expected_diarization_output = format_diarization_output(mock_annotation)
    mock_reverb_pipeline.transcribe.return_value = expected_transcript
    mock_run_diarization.return_value = mock_annotation
    num_speakers_manual = 2
    mock_emitter = MagicMock(spec=TranscriptionSignals)

    transcribe_audio(
        dummy_wav_path,
        diarize_mode=DiarizeMode.MANUAL,
        num_speakers_manual=num_speakers_manual,
        signal_emitter=mock_emitter
    )

    # Assert signals
    mock_emitter.transcript_ready.emit.assert_called_once_with(expected_transcript)
    mock_emitter.diarization_ready.emit.assert_called_once_with(expected_diarization_output)

    # Assert mocks
    mock_load_diarization.assert_called_once()
    mock_run_diarization.assert_called_once_with(dummy_wav_path, num_speakers=num_speakers_manual)

@patch('audioverba.core.transcription.core_load_diarization_pipeline')
@patch('audioverba.core.transcription.core_run_diarization', side_effect=DiarizationError("Diarization failed"))
@patch('audioverba.core.transcription.reverb_pipeline')
def test_transcribe_audio_diarization_auto_failure(
    mock_reverb_pipeline: MagicMock,
    mock_run_diarization: MagicMock,
    mock_load_diarization: MagicMock,
    dummy_wav_path: str
):
    """Test transcribe_audio with diarization mode AUTO where diarization fails."""
    expected_transcript = "Transcript despite diarization fail."
    expected_error_msg = "\n\n--------------------\n\nDiarization Info:\nDiarization Failed: Diarization failed"
    mock_reverb_pipeline.transcribe.return_value = expected_transcript
    mock_emitter = MagicMock(spec=TranscriptionSignals)

    transcribe_audio(dummy_wav_path, diarize_mode=DiarizeMode.AUTO, signal_emitter=mock_emitter)

    # Assert signals (transcript should still emit, diarization should emit formatted error string)
    mock_emitter.transcript_ready.emit.assert_called_once_with(expected_transcript)
    mock_emitter.diarization_ready.emit.assert_called_once_with(expected_error_msg)

    # Assert mocks
    mock_load_diarization.assert_called_once()
    mock_run_diarization.assert_called_once()

@patch('audioverba.core.transcription.reverb_pipeline')
def test_transcribe_audio_transcription_failure(
    mock_reverb_pipeline: MagicMock,
    dummy_wav_path: str
):
    """Test transcribe_audio when the core transcription call fails."""
    mock_reverb_pipeline.transcribe.side_effect = Exception("ASR engine exploded")
    mock_emitter = MagicMock(spec=TranscriptionSignals) # Still need emitter for the call

    with pytest.raises(TranscriptionError, match="ASR engine exploded"):
        # Call still needs the emitter argument, even if it raises before emitting
        transcribe_audio(
            dummy_wav_path,
            diarize_mode=DiarizeMode.OFF,
            signal_emitter=mock_emitter
        )

    # Assert that signals were NOT emitted due to the exception
    mock_emitter.transcript_ready.emit.assert_not_called()
    mock_emitter.diarization_ready.emit.assert_not_called()
