import pytest
from pathlib import Path

from audioverba.core.conversion import convert_to_wav, ConversionError

# Test function naming follows pytest conventions (test_*)

def test_convert_mp3_to_wav(test_audio_mp3: Path, tmp_path: Path):
    """Test converting an MP3 file to WAV."""
    assert test_audio_mp3.exists(), "Test MP3 file fixture should exist"
    
    try:
        output_wav_path = convert_to_wav(str(test_audio_mp3))
    except ConversionError as e:
        pytest.fail(f"Conversion failed unexpectedly: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during conversion: {e}")

    # Assertions
    assert output_wav_path is not None, "Output path should not be None"
    output_path_obj = Path(output_wav_path)
    assert output_path_obj.exists(), "Output WAV file should exist"
    assert output_path_obj.is_file(), "Output path should be a file"
    assert output_path_obj.suffix.lower() == ".wav", "Output file should have .wav extension"

def test_convert_mp4_to_wav(test_video_mp4: Path, tmp_path: Path):
    """Test converting an MP4 file to WAV."""
    assert test_video_mp4.exists(), "Test MP4 file fixture should exist"

    try:
        output_wav_path = convert_to_wav(str(test_video_mp4))
    except ConversionError as e:
        pytest.fail(f"Conversion failed unexpectedly: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during conversion: {e}")

    # Assertions
    assert output_wav_path is not None, "Output path should not be None"
    output_path_obj = Path(output_wav_path)
    assert output_path_obj.exists(), "Output WAV file should exist"
    assert output_path_obj.is_file(), "Output path should be a file"
    assert output_path_obj.suffix.lower() == ".wav", "Output file should have .wav extension"

def test_convert_non_existent_file(tmp_path: Path):
    """Test attempting to convert a file that does not exist."""
    non_existent_file = tmp_path / "does_not_exist.mp3"
    with pytest.raises(FileNotFoundError): # Expecting FileNotFoundError from ffmpeg check
        convert_to_wav(str(non_existent_file))

def test_convert_unsupported_file(tmp_path: Path):
    """Test attempting to convert an unsupported file type (e.g., a text file)."""
    unsupported_file = tmp_path / "test.txt"
    unsupported_file.write_text("This is not audio.")
    
    with pytest.raises(ConversionError): # Expecting a conversion error
        convert_to_wav(str(unsupported_file))
