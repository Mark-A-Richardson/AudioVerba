import os
import logging
from typing import Optional
from PySide6.QtCore import QObject, Signal, Slot, QRunnable, QThreadPool, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QCheckBox, QHBoxLayout,
    QSpinBox, QTextEdit, QPushButton, QFileDialog, QProgressBar
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent # Add QMouseEvent here

from ..core.conversion import convert_to_wav, ConversionError # Corrected import path
from ..core.transcription import (
    TranscriptionError, transcribe_audio,
    load_reverb_model, DiarizeMode, TranscriptionSignals # Corrected import name
)
from ..core.diarization import DiarizationError # Import the specific error

# Setup logging
log = logging.getLogger(__name__)

# --- Worker Signals --- #
class WorkerSignals(QObject):
    ''' Defines the signals available from a running worker thread. '''
    progress = Signal(int, str)  # Signal to emit progress updates (text)
    # Signal to emit results (tuple: transcript, diarization_info)
    result = Signal(object)
    error = Signal(tuple)   # Signal to emit errors (type, value, traceback tuple)
    finished = Signal()     # Signal indicating the thread has finished

# --- Worker Thread --- #
class Worker(QRunnable):
    """Worker thread for handling audio processing."""
    def __init__(self, file_path: str, diarize_mode: DiarizeMode, num_speakers: Optional[int]):
        super().__init__()
        self.file_path = file_path
        self.diarize_mode = diarize_mode
        self.num_speakers = num_speakers
        self.signals = WorkerSignals()

        # Temporary storage for results received via internal signals
        self._received_transcript: Optional[str] = None
        self._received_diarization: Optional[object] = None # str or None

    # --- Internal Slots --- #
    # Using simple methods for clarity over lambdas here
    @Slot(str)
    def _handle_transcript(self, transcript: str):
        log.debug(f"WORKER INTERNAL SIGNAL: Received transcript (len: {len(transcript)})")
        self._received_transcript = transcript

    @Slot(object)
    def _handle_diarization(self, diarization_data: object):
        log.debug(f"WORKER INTERNAL SIGNAL: Received diarization (type: {type(diarization_data).__name__})")
        self._received_diarization = diarization_data

    @Slot()
    def run(self) -> None:
        """Execute the audio processing task."""
        log.info(f"Worker started for: {self.file_path}")
        wav_path = None
        final_transcript = ""
        final_diarization = None

        internal_emitter = TranscriptionSignals() # Create the internal signal emitter

        # --- Connect internal signals --- #
        # Connections are critical, so let errors propagate if they fail here
        internal_emitter.transcript_ready.connect(self._handle_transcript)
        internal_emitter.diarization_ready.connect(self._handle_diarization)
        log.debug("Worker(QRunnable) internal signals connected successfully.")

        try:
            self.signals.progress.emit(10, "Converting to WAV...")
            try:
                log.debug("Worker(QRunnable): Calling convert_to_wav...")
                wav_path = convert_to_wav(self.file_path)
                log.info(f"Successfully converted to WAV: {wav_path}")
            except ConversionError as e:
                log.error(f"Conversion failed: {e}", exc_info=True)
                self.signals.error.emit(f"Conversion Error: {e}")
                return # Stop processing if conversion fails
            except Exception as e:
                log.error(f"Unexpected error during conversion: {e}", exc_info=True)
                self.signals.error.emit(f"Unexpected Conversion Error: {e}")
                return

            self.signals.progress.emit(50, "Transcribing...")

            # --- Call transcribe_audio (now doesn't return directly) --- #
            log.debug("Worker(QRunnable): Calling transcribe_audio...")
            transcribe_audio(
                wav_path,
                signal_emitter=internal_emitter, # Pass the emitter
                diarize_mode=self.diarize_mode,
                num_speakers_manual=self.num_speakers,
            )
            log.debug("Worker(QRunnable): transcribe_audio call finished.")

            # --- Assemble result from internal signal slots --- #
            log.debug("Worker(QRunnable): Assembling final result...")
            # Ensure default values if signals didn't fire (e.g., due to early error)
            final_transcript = self._received_transcript if self._received_transcript is not None else ""
            final_diarization = self._received_diarization # Can be str or None

            log.info("Transcription complete.")
            log.debug(f"Worker(QRunnable) preparing to emit final data: Type={type((final_transcript, final_diarization)).__name__}, Value Snippet: {str((final_transcript, final_diarization))[:100]}...")
            self.signals.result.emit((final_transcript, final_diarization))

        except (TranscriptionError, DiarizationError) as e:
            # Handle errors from transcribe_audio (e.g., model loading failures)
            log.error(f"Processing error in worker: {e}", exc_info=True)
            self.signals.error.emit(f"Processing Error: {e}")
        except Exception as e:
            log.exception("An unexpected error occurred in the worker thread") # Log full traceback
            self.signals.error.emit(f"An unexpected error occurred: {e}")
        finally:
            # --- Disconnect internal signals (good practice) --- #
            try:
                log.debug("Worker(QRunnable): Disconnecting internal signals...")
                internal_emitter.transcript_ready.disconnect(self._handle_transcript)
                internal_emitter.diarization_ready.disconnect(self._handle_diarization)
                log.debug("Worker(QRunnable): Internal signals disconnected.")
            except RuntimeError as e:
                 # Can happen if signals were already disconnected or object destroyed
                 log.warning(f"Could not disconnect worker internal signals: {e}")

            self.signals.finished.emit()
            log.info("Worker finished.")
            if wav_path and wav_path != self.file_path:
                try:
                    log.debug(f"Attempting to delete temporary WAV file: {wav_path}")
                    os.remove(wav_path)
                    log.info(f"Successfully deleted temporary WAV file: {wav_path}")
                except OSError as e_del:
                    log.warning(f"Could not delete temporary WAV file {wav_path}: {e_del}")

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
        self._selected_file_path: Optional[str] = None # Store the actual file path

        self.setup_ui()
        self.load_models()

    def setup_ui(self) -> None:
        """Sets up the user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Drop Zone Label
        self.drop_zone_label = QLabel("Drag & Drop Audio/Video File Here\n(or click to browse)")
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
        self.speaker_count_spinbox.setEnabled(False) # Explicitly disable initially
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

        # Process Button
        self.process_button = QPushButton("Process Audio")
        self.process_button.clicked.connect(self._start_processing) # Connect to the processing method
        self.process_button.setEnabled(False) # Initially disabled
        main_layout.addWidget(self.process_button)

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
        """Handles dropped files."""
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            url = mime_data.urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                # Basic check for likely audio/video files
                allowed_extensions = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".mp4", ".avi", ".mov", ".mkv")
                if file_path.lower().endswith(allowed_extensions):
                    self._handle_valid_file_selected(file_path)
                else:
                    log.warning(f"Dropped file has unsupported extension: {file_path}")
                    self.status_label.setText("Unsupported file type.")
                event.acceptProposedAction()
            else:
                log.warning(f"Dropped item is not a local file: {url}")
                self.status_label.setText("Please drop a local file.")
                event.acceptProposedAction()
        else:
            super().dropEvent(event)

    @Slot()
    def browse_file(self) -> None:
        """Opens a file dialog to select an audio/video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio/Video File",
            "", # Start directory
            "Audio/Video Files (*.wav *.mp3 *.ogg *.flac *.m4a *.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self._handle_valid_file_selected(file_path)

    def _handle_valid_file_selected(self, file_path: str) -> None:
        """Updates UI elements after a valid audio/video file is selected."""
        self._selected_file_path = file_path # Store the actual path
        base_name = os.path.basename(file_path)
        self.drop_zone_label.setText(f"Selected: {base_name}") # Update display
        log.info(f"Valid file selected: {file_path}")
        self.process_button.setEnabled(True) # Enable process button
        self.export_button.setEnabled(False) # Disable export until processed
        self.transcript_output.clear()
        self.diarization_output.clear()
        self.diarization_output_label.setVisible(False)
        self.diarization_output.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("File selected. Click 'Process Audio'.")

    # --- Processing Logic --- #
    @Slot()
    def _start_processing(self) -> None:
        """Starts the audio processing worker thread."""
        file_path = self._selected_file_path # Use stored path
        if not file_path or not os.path.exists(file_path):
            log.warning("Process button clicked with no valid file selected.")
            self.status_label.setText("Please select a valid file first.")
            self.process_button.setEnabled(False) # Ensure it's disabled if file is invalid
            return

        log.info(f"Process button clicked for file: {file_path}")
        self.drop_zone_label.setText(f"Processing: {os.path.basename(file_path)}")

        # Get settings before starting worker
        diarize_enabled = self.diarize_checkbox.isChecked()
        speaker_count_value = self.speaker_count_spinbox.value()

        # Determine DiarizeMode
        if diarize_enabled:
            # Simplified: AUTO if spinbox is 0, MANUAL otherwise (consistent with spinbox enable logic)
            diarization_mode = DiarizeMode.MANUAL if speaker_count_value > 0 else DiarizeMode.AUTO
        else:
            diarization_mode = DiarizeMode.OFF

        # Use None if mode is not MANUAL
        speakers_param = speaker_count_value if diarization_mode == DiarizeMode.MANUAL else None

        # --- Disable controls and clear outputs --- #
        log.debug("Disabling UI controls for processing.")
        self.diarize_checkbox.setEnabled(False)
        self.speaker_count_spinbox.setEnabled(False)
        self.process_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.transcript_output.clear()
        self.diarization_output.clear()
        self.diarization_output_label.setVisible(False)
        self.diarization_output.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.last_transcript = "" # Reset last transcript
        self.last_diarization_info = None # Reset last diarization

        # --- Start processing in a separate thread --- #
        log.info(f"Creating worker with file: {file_path}, mode: {diarization_mode}, speakers: {speakers_param}")
        self.worker = Worker(file_path, diarization_mode, speakers_param)

        # Connect worker signals to main window slots
        self.worker.signals.progress.connect(self.update_status_label)
        self.worker.signals.result.connect(self.processing_finished)
        self.worker.signals.error.connect(self.processing_error)
        self.worker.signals.finished.connect(self._enable_controls) # Re-enable controls when finished

        # Start the worker thread using the global thread pool
        QThreadPool.globalInstance().start(self.worker)
        log.info("Worker thread submitted to pool.")

    @Slot(int, str)
    def update_status_label(self, progress: int, message: str) -> None:
        """Updates the status label and progress bar."""
        # Log verbosely for debugging, maybe reduce later
        log.debug(f"Progress Update: {progress}% - {message}")
        self.status_label.setText(message)
        self.progress_bar.setValue(progress)

    @Slot(int)
    def toggle_speaker_count(self, state: int) -> None:
        """Shows/hides AND enables/disables the speaker count spinbox based on diarize checkbox state."""
        # state will be Qt.CheckState enum (e.g., Qt.Checked, Qt.Unchecked)
        is_checked = (Qt.CheckState(state) == Qt.Checked)
        self.speaker_count_label.setVisible(is_checked)
        self.speaker_count_spinbox.setVisible(is_checked)
        self.speaker_count_spinbox.setEnabled(is_checked) # Also set enabled state
        log.debug(f"Speaker count visibility and enabled state toggled: {is_checked}")

    # --- File Handling --- #
    @Slot()
    def export_transcript(self) -> None:
        """Exports the displayed transcript to a text file."""
        if not self.last_transcript:
            log.warning("Export attempted with no transcript available.")
            self.status_label.setText("No transcript to export.")
            return

        original_file_path = self._selected_file_path # Use stored path
        suggested_name = "transcript.txt"
        if original_file_path: # Check if a path was actually stored
            try:
                base_name = os.path.basename(original_file_path)
                suggested_name = os.path.splitext(base_name)[0] + ".txt"
            except Exception as e:
                log.warning(f"Could not generate suggested filename from {original_file_path}: {e}")

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Transcript", suggested_name, "Text Files (*.txt);;All Files (*)"
        )

        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(self.last_transcript)
                    if self.last_diarization_info:
                        f.write("\n\n--- Speaker Timeline ---\n")
                        f.write(self.last_diarization_info)
                log.info(f"Transcript exported successfully to: {save_path}")
                self.status_label.setText("Transcript exported.")
            except Exception as e:
                log.error(f"Failed to export transcript: {e}", exc_info=True)
                self.status_label.setText(f"Error exporting transcript: {e}")
        else:
            log.info("Export cancelled by user.")
            self.status_label.setText("Export cancelled.")

    @Slot(object)
    def processing_finished(self, result: object) -> None:
        """Handles the result from the processing thread."""
        log.info(f"Processing finished signal received. Result type: {type(result)}")
        transcript: Optional[str] = None
        diarization_display_text: str = ""

        if isinstance(result, tuple) and len(result) == 2:
            transcript, diarization_info = result # Expecting (str, str | None)
            log.info(f"Received Transcript length: {len(transcript) if transcript else 0}")
            if isinstance(diarization_info, str):
                log.info(f"Received Diarization Info string (length: {len(diarization_info)})")
                diarization_display_text = diarization_info
            elif diarization_info is None:
                log.info("Received Diarization Info: None")
            else:
                log.warning(f"Received unexpected type for Diarization Info: {type(diarization_info).__name__}")
                diarization_display_text = f"Error: Unexpected diarization result type ({type(diarization_info).__name__})"
        else:
            log.warning(f"Received unexpected result format: {type(result).__name__}")
            transcript = f"Error: Unexpected result format ({type(result).__name__})"

        self.status_label.setText("Processing finished.")
        self.progress_bar.setValue(100)
        self.drop_zone_label.setText("Drag & Drop Audio/Video File Here\n(or click to browse)")

        # Store and display results
        self.last_transcript = transcript
        self.last_diarization_info = diarization_display_text if diarization_display_text else None

        self.transcript_output.setPlainText(transcript if transcript else "")

        if diarization_display_text:
            log.debug("Setting diarization_output text...")
            self.diarization_output.setPlainText(diarization_display_text)
            self.diarization_output_label.setVisible(True)
            self.diarization_output.setVisible(True)
        else:
            self.diarization_output.clear()
            self.diarization_output_label.setVisible(False)
            self.diarization_output.setVisible(False)

        # Button enabling is handled by _enable_controls via finished signal

    @Slot()
    def processing_error(self, error: tuple) -> None:
        """Handles errors from the processing thread."""
        error_type, error_value, error_traceback = error
        log.error(f"Processing error: {error_type.__name__}: {error_value}", exc_info=error_traceback)
        self.status_label.setText(f"Error: {error_type.__name__}: {error_value}")
        self.progress_bar.setValue(0)
        self.drop_zone_label.setText("Drag & Drop Audio/Video File Here\n(or click to browse)")

    @Slot()
    def _enable_controls(self) -> None:
        """Re-enables UI controls after processing is finished or stopped."""
        log.debug("Re-enabling UI controls.")
        self.diarize_checkbox.setEnabled(True)
        self.speaker_count_spinbox.setEnabled(self.diarize_checkbox.isChecked())
        self.process_button.setEnabled(True)
        self.export_button.setEnabled(bool(self.transcript_output.toPlainText()))
        self.worker = None

    # ... (export_transcript)

    # --- Drag and Drop --- #
    # ... (dragEnterEvent, dragLeaveEvent, dropEvent)

    # --- Window Events --- #
    # ... (mousePressEvent, closeEvent)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle clicks, specifically on the drop zone label to trigger browse."""
        if event.button() == Qt.MouseButton.LeftButton and self.drop_zone_label.geometry().contains(event.pos()):
            log.debug("Drop zone clicked, triggering file browse.")
            self.browse_file()
            event.accept() # Consume the event
        else:
            super().mousePressEvent(event) # Pass on other clicks
