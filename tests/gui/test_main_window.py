import pytest
import os
from pytestqt.plugin import QtBot
from audioverba.gui.main_window import MainWindow

# Fixture to create the main window instance for each test
@pytest.fixture(scope="function") # Changed scope to function for isolation
def main_window(qtbot) -> MainWindow:
    """Create an instance of the main window."""
    # Note: We might need to mock model loading here in the future
    # if it's too slow or requires network access for tests.
    window = MainWindow()
    qtbot.addWidget(window) # Ensure qtbot manages the widget lifecycle
    window.show() # Make sure window is shown for visibility tests
    return window

def test_initial_state(qtbot, main_window: MainWindow):
    """Test the initial state of the main window widgets."""
    assert main_window.isVisible()
    assert main_window.drop_zone_label is not None # Check if earlier widget exists
    assert not main_window.process_button.isEnabled() # Process button should be initially disabled
    assert main_window.transcript_output.toPlainText() == "" # Transcription area empty
    assert main_window.diarization_output.toPlainText() == "" # Diarization area empty
    # Check initial drop zone text (adjust if needed)
    assert main_window.drop_zone_label.text() == "Drag & Drop Audio/Video File Here\n(or click to browse)"

    # Check initial state of diarization controls
    assert not main_window.diarize_checkbox.isChecked() # Diarize checkbox off by default
    assert not main_window.speaker_count_label.isVisible() # Speaker label hidden
    assert not main_window.speaker_count_spinbox.isVisible() # Speaker spinbox hidden
    assert main_window.speaker_count_spinbox.value() == 0 # Speaker count is 0

    # Check initial state of export button
    assert not main_window.export_button.isEnabled() # Export button disabled initially

def test_file_selection_enables_process_button(qtbot, main_window: MainWindow):
    """Test that selecting a valid file enables the process button and updates the label."""
    # Initially, the process button should be disabled
    assert not main_window.process_button.isEnabled()

    # Simulate selecting a valid file
    dummy_file_path = "C:/path/to/dummy_audio.wav" # Use forward slashes for cross-platform compatibility
    expected_base_name = os.path.basename(dummy_file_path)
    expected_label_text = f"Selected: {expected_base_name}"

    # Call the internal handler directly for testing
    main_window._handle_valid_file_selected(dummy_file_path)

    # Assertions after file selection
    assert main_window.process_button.isEnabled()
    assert main_window.drop_zone_label.text() == expected_label_text
    assert main_window._selected_file_path == dummy_file_path # Check internal state too
    assert main_window.status_label.text() == "File selected. Click 'Process Audio'."
    assert not main_window.export_button.isEnabled() # Export button should still be disabled

def test_toggle_diarization_controls(qtbot: QtBot, main_window: MainWindow):
    """Test toggling the diarization checkbox shows/hides/enables/disables speaker controls."""
    # Initial state checks (redundant with test_initial_state but good for clarity)
    assert not main_window.diarize_checkbox.isChecked()
    assert not main_window.speaker_count_label.isVisible()
    assert not main_window.speaker_count_spinbox.isVisible()
    assert not main_window.speaker_count_spinbox.isEnabled()

    # Simulate checking the checkbox
    main_window.diarize_checkbox.setChecked(True)
    # Alternative using qtbot signals if needed: qtbot.click(main_window.diarize_checkbox)

    # Assert controls are visible and enabled after checking
    assert main_window.speaker_count_label.isVisible()
    assert main_window.speaker_count_spinbox.isVisible()
    assert main_window.speaker_count_spinbox.isEnabled()

    # Simulate unchecking the checkbox
    main_window.diarize_checkbox.setChecked(False)

    # Assert controls are hidden and disabled again after unchecking
    assert not main_window.speaker_count_label.isVisible()
    assert not main_window.speaker_count_spinbox.isVisible()
    assert not main_window.speaker_count_spinbox.isEnabled()
