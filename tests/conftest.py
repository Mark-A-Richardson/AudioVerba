import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the root directory of the project."""
    # Assumes conftest.py is in the 'tests' directory, one level below root
    return Path(__file__).parent.parent.resolve()

@pytest.fixture(scope="session")
def test_audio_mp3(project_root: Path) -> Path:
    """Return the path to the test MP3 audio file."""
    file_path = project_root / "Test_Audio.mp3"
    if not file_path.is_file():
        pytest.skip(f"Test audio file not found: {file_path}")
    return file_path

@pytest.fixture(scope="session")
def test_video_mp4(project_root: Path) -> Path:
    """Return the path to the test MP4 video file."""
    file_path = project_root / "Test_Video.mp4"
    if not file_path.is_file():
        pytest.skip(f"Test video file not found: {file_path}")
    return file_path
