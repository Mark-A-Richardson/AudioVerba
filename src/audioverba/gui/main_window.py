import sys
import os
import logging
from typing import Optional, Tuple

from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QCheckBox, QHBoxLayout,
    QSpinBox, QTextEdit, QPushButton, QFileDialog, QProgressBar
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from ..core.conversion import convert_to_wav, ConversionError # Corrected import path
from ..core.transcription import (
    TranscriptionError, transcribe_audio,
    load_reverb_model, DiarizeMode, format_diarization_output # Corrected import name
)
from ..core.diarization import DiarizationError # Import the specific error
from pyannote.core import Annotation # Import Annotation type

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Worker Thread --- #
class WorkerSignals(QObject):
    ''' Defines the signals available from a running worker thread. '''
    progress = Signal(str)  # Signal to emit progress updates (text)
    result = Signal(object) # Signal to emit results (tuple: transcript, diarization_info)
    error = Signal(tuple)   # Signal to emit errors (type, value, traceback tuple)
    finished = Signal()     # Signal indicating the thread has finished

class Worker(QThread):
    ''' Generic worker thread '''
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        ''' Initialise the runner function with passed args, kwargs. '''
        # Ensure logging is configured for this thread
        logger = logging.getLogger() # Get root logger
        logger.info(f"Worker {self.currentThread()}: Starting run method.")
        # Retrieve args/kwargs here
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            logger.exception(f"Worker {self.currentThread()}: Error during execution.")
            import traceback # Ensure traceback is available here
            traceback_str = traceback.format_exc()
            log.error(f"Worker thread error: {traceback_str}")
            self.signals.error.emit(sys.exc_info())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

# --- Main Window --- #
class MainWindow(QMainWindow):
    """Main application window for AudioVerba."""

    def __init__(self) -> None:
        """Initializes the main window."""
        super().__init__()
        self.setWindowTitle("AudioVerba")
        self.setGeometry(100, 100, 700, 550) # Adjusted size
        self.setAcceptDrops(True) # Enable drag and drop

        self.input_file_path: Optional[str] = None
        self.last_processed_wav_path: Optional[str] = None
        self.last_transcript: Optional[str] = None
        self.last_diarization_info: Optional[str] = None
        self.worker: Optional[Worker] = None

        self.setup_ui()
        self.load_models()

    def setup_ui(self) -> None:
        """Sets up the user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Drop Zone Label
        self.drop_zone_label = QLabel("Drag & Drop Audio/Video File Here")
        self.drop_zone_label.setStyleSheet("QLabel { border: 2px dashed #aaa; padding: 20px; text-align: center; font-size: 14px; }" )
        self.drop_zone_label.setMinimumHeight(80)
        self.drop_zone_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.drop_zone_label)

        # Settings Layout
        settings_layout = QHBoxLayout()
        self.diarize_checkbox = QCheckBox("Enable Speaker Diarization")
        self.diarize_checkbox.setToolTip("Identify different speakers. Set count to 0 for auto-detect.")
        self.speaker_count_label = QLabel("Speaker Count (0=Auto):")
        self.speaker_count_spinbox = QSpinBox()
        self.speaker_count_spinbox.setRange(0, 10)
        self.speaker_count_spinbox.setValue(0)
        self.speaker_count_label.setVisible(False)
        self.speaker_count_spinbox.setVisible(False)
        self.diarize_checkbox.stateChanged.connect(self.toggle_speaker_count)
        settings_layout.addWidget(self.diarize_checkbox)
        settings_layout.addWidget(self.speaker_count_label)
        settings_layout.addWidget(self.speaker_count_spinbox)
        settings_layout.addStretch(1)
        main_layout.addLayout(settings_layout)

        # Transcript Output Area
        self.transcript_output = QTextEdit()
        self.transcript_output.setReadOnly(True)
        self.transcript_output.setPlaceholderText("Transcript will appear here...")
        main_layout.addWidget(self.transcript_output, 1)

        # Diarization Output Area
        self.diarization_output_label = QLabel("Speaker Timeline:")
        self.diarization_output_label.setVisible(False)
        self.diarization_output = QTextEdit()
        self.diarization_output.setReadOnly(True)
        self.diarization_output.setFixedHeight(100)
        self.diarization_output.setVisible(False)
        main_layout.addWidget(self.diarization_output_label)
        main_layout.addWidget(self.diarization_output)

        # Progress Bar & Status Label
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

        # Export Button Layout
        export_layout = QHBoxLayout()
        export_layout.addStretch(1)
        self.export_button = QPushButton("Export Transcript (.txt)")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_transcript)
        export_layout.addWidget(self.export_button)
        main_layout.addLayout(export_layout)

    # --- Model Loading --- #
    def load_models(self) -> None:
        """Loads ASR and potentially diarization models at startup."""
        self.status_label.setText("Loading ASR model...")
        QApplication.processEvents() # Update UI
        try:
            load_reverb_model()
            log.info("Reverb ASR model loaded successfully.")
            self.status_label.setText("Models loaded. Ready.")
        except TranscriptionError as e:
            log.error(f"Failed to load ASR model: {e}")
            self.status_label.setText(f"Error loading ASR model: {e}")
        # Diarization model is loaded on demand by the core module

    # --- UI Event Handlers --- #
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        mime_data = event.mimeData()
        if mime_data.hasUrls() and len(mime_data.urls()) == 1:
            url = mime_data.urls()[0]
            if url.isLocalFile():
                self.input_file_path = url.toLocalFile()
                self._process_selected_file(self.input_file_path)
            else:
                event.ignore()
        else:
            event.ignore()

    def mousePressEvent(self, event) -> None:
        """Handle clicks, specifically on the drop zone label to trigger browse."""
        if event.button() == Qt.LeftButton and self.drop_zone_label.geometry().contains(event.pos()):
            self.browse_file()
        else:
            super().mousePressEvent(event) # Pass on other clicks

    @Slot(int)
    def toggle_speaker_count(self, state: int) -> None:
        """Shows/hides the speaker count spinbox based on diarize checkbox state."""
        is_enabled = bool(state)
        self.speaker_count_label.setVisible(is_enabled)
        self.speaker_count_spinbox.setVisible(is_enabled)
        self.speaker_count_spinbox.setEnabled(is_enabled) # Also control enabled state
        if not is_enabled:
            self.speaker_count_spinbox.setValue(0) # Reset to auto when disabled

    @Slot()
    def browse_file(self) -> None:
        """Opens a file dialog to select an input audio/video file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio/Video File")
        if file_path:
            log.info(f"File selected via browse: {file_path}")
            self._process_selected_file(file_path)

    @Slot()
    def export_transcript(self) -> None:
        """Exports the displayed transcript to a text file."""
        if not self.last_transcript:
            log.warning("Export attempted with no transcript available.")
            self.status_label.setText("No transcript to export.")
            return

        # Suggest filename based on original input
        if self.input_file_path:
            base_name = os.path.basename(self.input_file_path)
            suggested_name = os.path.splitext(base_name)[0] + ".txt"
        else:
            suggested_name = "transcript.txt"

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Transcript", suggested_name, "Text Files (*.txt)")

        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(self.last_transcript)
                log.info(f"Transcript exported successfully to: {save_path}")
                self.status_label.setText("Transcript exported.")
            except Exception as e:
                log.error(f"Failed to export transcript: {e}")
                self.status_label.setText(f"Error exporting transcript: {e}")

    # --- Worker Target & Slots --- #
    def _process_selected_file(self, file_path: str) -> None:
        """Common logic to handle a selected file (from drop or browse)."""
        if self.worker is not None and self.worker.isRunning():
            log.warning("Processing already in progress. Ignoring new file selection.")
            self.status_label.setText("Processing already in progress...")
            return

        self.input_file_path = file_path # Store the path being processed
        log.info(f"Processing requested for: {file_path}")
        self.drop_zone_label.setText(f"Processing: {os.path.basename(file_path)}")

        # Get settings before starting worker
        diarize_enabled = self.diarize_checkbox.isChecked()
        speaker_count_value = self.speaker_count_spinbox.value()

        # Determine DiarizeMode
        if diarize_enabled:
            diarization_mode = DiarizeMode.MANUAL if speaker_count_value > 0 else DiarizeMode.AUTO
        else:
            diarization_mode = DiarizeMode.OFF

        # Use None if mode is not MANUAL
        speakers_param = speaker_count_value if diarization_mode == DiarizeMode.MANUAL else None

        # Disable widgets during processing
        self.diarize_checkbox.setEnabled(False)
        self.speaker_count_spinbox.setEnabled(False)
        self.export_button.setEnabled(False)
        self.transcript_output.clear()
        self.diarization_output.clear()
        self.diarization_output_label.setVisible(False)
        self.diarization_output.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")

        # Start processing in a separate thread
        self.worker = Worker(self.start_processing, file_path, diarization_mode, speakers_param)
        self.worker.signals.progress.connect(self.update_status_label)
        self.worker.signals.result.connect(self.processing_finished)
        self.worker.signals.error.connect(self.processing_error)
        # finished signal is connected implicitly to clean up worker later if needed
        self.worker.start()

    def start_processing(self, input_file: str, diarize_mode: DiarizeMode, num_speakers: Optional[int]) -> Tuple[str, Optional[str]]:
        """Runs the conversion and transcription/diarization pipeline. (Executed by Worker)"""
        # Emit signals via the worker instance stored in MainWindow
        if self.worker:
            self.worker.signals.progress.emit("Starting conversion...")
        try:
            converted_wav_path = convert_to_wav(input_file)
            if self.worker:
                self.worker.signals.progress.emit(f"Converted to: {os.path.basename(converted_wav_path)}")
            self.last_processed_wav_path = converted_wav_path # Store for potential debugging

            if self.worker:
                self.worker.signals.progress.emit("Starting transcription...")
            # Correctly call transcribe_audio with keyword arguments
            transcript_text, diarization_result = transcribe_audio(
                converted_wav_path, # Pass path positionally
                diarize_mode=diarize_mode,
                num_speakers_manual=num_speakers
            )
            if self.worker:
                self.worker.signals.progress.emit("Transcription complete.")
            
            # Format the output string
            final_output = f"Transcript:\n{transcript_text}\n"
            final_output += "\n" + "-"*20 + "\n\n"

            # Check the type of diarization_result
            if isinstance(diarization_result, Annotation):
                formatted_diarization = format_diarization_output(diarization_result)
                final_output += f"Speaker Timeline:\n{formatted_diarization}"
            elif isinstance(diarization_result, str):
                # It's likely an error message or 'Diarization disabled'
                final_output += f"Diarization Info:\n{diarization_result}"
            elif diarization_result is None:
                # Diarization was off or returned nothing
                final_output += "Diarization Info:\nDiarization was not enabled."
            else:
                # Should not happen based on transcribe_audio return type
                final_output += f"Diarization Info:\nUnexpected data type received: {type(diarization_result)}"

            return final_output, None # Return the combined output
        except (ConversionError, TranscriptionError, DiarizationError) as e:
            error_message = f"Processing error: {e}"
            log.exception(error_message) # Log full traceback
            # Reraise the specific error to be caught by the worker's run method
            raise
        except Exception as e:
            # Catch-all for unexpected errors during the process
            error_message = f"Unexpected processing error: {e}"
            log.exception(error_message)
            # Wrap in a known error type if desired, or re-raise
            raise TranscriptionError(f"Unexpected error: {e}") # Wrap as TranscriptionError

    @Slot(str)
    def update_status_label(self, message: str) -> None:
        """Updates the status label and logs the message."""
        log.info(message) # Log progress messages
        self.status_label.setText(message)
        # Update progress bar based on message - simplistic approach
        if "conversion" in message.lower():
            self.progress_bar.setValue(25)
        elif "converted" in message.lower():
            self.progress_bar.setValue(50)
        elif "transcription" in message.lower():
            self.progress_bar.setValue(75)
        elif "complete" in message.lower():
            self.progress_bar.setValue(100)

    @Slot(object)
    def processing_finished(self, result: Tuple[str, Optional[str]]) -> None:
        """Handles the result from the processing thread."""
        log.info(f"Processing finished. Received result: {type(result)}")
        if isinstance(result, tuple) and len(result) == 2:
            transcript_log, diarization_info_log = result
            log.info(f"Received Transcript length: {len(transcript_log) if transcript_log else 0}")
            log.info(f"Received Diarization Info: {diarization_info_log}") # Log the received diarization data
        else:
            log.warning(f"Received unexpected result format: {result}")

        self.status_label.setText("Processing finished.")
        self.progress_bar.setValue(100)
        self.drop_zone_label.setText("Drag & Drop Audio/Video File Here")

        transcript, diarization_info = result
        self.last_transcript = transcript
        self.last_diarization_info = diarization_info

        # Re-enable widgets after processing
        self.diarize_checkbox.setEnabled(True)
        self.speaker_count_spinbox.setEnabled(self.diarize_checkbox.isChecked())

        # Display the transcript
        self.transcript_output.setPlainText(transcript if transcript else "")
        self.export_button.setEnabled(bool(transcript))

        # Display diarization info if available
        if diarization_info:
            self.diarization_output.setPlainText(diarization_info)
            self.diarization_output_label.setVisible(True)
            self.diarization_output.setVisible(True)
        else:
            self.diarization_output.clear()
            self.diarization_output_label.setVisible(False)
            self.diarization_output.setVisible(False)

        self.worker = None # Clear worker reference

    @Slot(tuple)
    def processing_error(self, exception_tuple: tuple) -> None:
        """Handles errors reported by the worker thread's 'error' signal."""
        error_type, error_value, _ = exception_tuple
        error_message = f"Error: {error_value}"
        log.error(f"Worker thread error signal received: {error_message}")
        self.status_label.setText(error_message)
        self.progress_bar.setValue(100) # Or reset to 0?
        self.transcript_output.clear()
        self.export_button.setEnabled(False)
        self.diarization_output.clear()
        self.diarization_output_label.setVisible(False)
        self.diarization_output.setVisible(False)

        # Re-enable widgets
        self.diarize_checkbox.setEnabled(True)
        self.speaker_count_spinbox.setEnabled(self.diarize_checkbox.isChecked())

        self.worker = None # Clear worker reference
