"""Custom exceptions for the audioverba core modules."""

class AudioVerbaError(Exception):
    """Base class for exceptions in this module."""
    pass

class TranscriptionError(AudioVerbaError):
    """Custom exception for transcription errors."""
    pass

class DiarizationError(AudioVerbaError):
    """Custom exception for diarization errors."""
    pass

class ConversionError(AudioVerbaError):
    """Custom exception for audio conversion errors."""
    pass
