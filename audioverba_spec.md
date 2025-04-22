# AudioVerba Specification

A standalone, GPU-accelerated desktop application for transcribing audio/video into plain text, powered by Reverb’s “large” ASR model and speaker diarization. Users get a lightweight, stylish PySide6 GUI that handles everything—from format conversion through denoising to transcription—without any external installs.

---

## 1. High-Level Architecture

```text
[User GUI: AudioVerba]
   └─► Import File
          ├─► Conversion (FFmpeg)
          └─► Transcription + Diarization (Reverb large model on GPU)
                   └─► Plain-text export (.txt)
```  

- **Language:** Python 3.10+  
- **Framework:** PySide6  
- **Bundled Tools:**  
  - FFmpeg (static binary)  
  - Reverb SDK & large-model weights  

---

## 2. Core Components & Workflow

| Step               | Description                                                                                                   | UI Element                                    |
|--------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| **1. Import**      | User drags/drops a file or clicks “Import”. Accepts audio/video formats (.mp4, .mov, .wav, .mp3, etc.).      | File-picker dialog + drag-drop zone           |
| **2. Conversion**  | Run FFmpeg → 16 kHz, 16-bit PCM, mono WAV.                                                                   | “Conversion” progress bar (0–100%)            |
| **3. Transcribe**  | Reverb large model with hidden best-practice parameters (beam width, LM weight, length penalty tuned for accuracy/speed trade-off). | “Transcription” sub-progress bar        |
| **4. Export**      | Save raw transcript as `.txt` (filename defaults to `<orig_name>.txt`).                                       | Auto-populated “Save” dialog                  |

---

## 3. Settings & Configuration

### 3.1 Exposed UI Settings

- **Speaker count (diarization):**  
  - ☐ Auto-detect (default)  
  - ○ Manual override (1–10)  
- **Output location & filename:**  
  - Folder selector  
  - Filename input (default = `<orig_name>.txt`)

### 3.2 Hidden Best-Practice Defaults

*(not user-editable)*

| Parameter            | Value / Rationale                         |
|----------------------|-------------------------------------------|
| Sample rate          | 16 kHz (Reverb-recommended)               |
| Bit depth            | 16-bit PCM, mono (optimal for ASR)        |
| Model variant        | Reverb “large” (highest accuracy)         |
| Beam width           | 12 (accuracy bias, minimal slowdown)      |
| LM weight            | 1.5 (more weight on language model)       |
| Length penalty       | 0.6 (discourage overly short outputs)     |

### 3.3 Persisting User Preferences

- Config path: `%APPDATA%/AudioVerba/config.json`.  
- On startup, read and prefill UI settings from this file; fallback to defaults if missing.  
- After each transcription run, overwrite with last-used settings.

---

## 4. GUI Design

### 4.1 Main Window

- **Header bar:** app name “AudioVerba”, “Check for updates” button  
- **Central pane:**  
  - Icon + “Drag & drop or click to import”  
  - After import: file metadata (name, duration)  
- **Accordion sidebar:** settings panels  
- **Footer:** two-segment progress bar + “Start” / “Cancel” buttons

### 4.2 Progress Bar Behavior

1. **Segment 1 (Conversion):** % of FFmpeg task  
2. **Segment 2 (Transcription):** advances based on Reverb callbacks  
- **Log console (collapsible):** real-time stdout/stderr tail

---

## 5. Packaging & Deployment

- **Installer (.exe/.msi):** bundles Python 3.8+, PySide6, Reverb SDK, FFmpeg.  
- Installs to `C:\Program Files\AudioVerba\` by default.  
- Creates Start Menu & desktop shortcut.  
- **Auto-Updater:** checks GitHub Releases RSS; prompts user to download/install updates.  
- **Uninstaller:** standard Add/Remove Programs entry.

---

## 6. Error Handling & Logging

- **Pre-run validation:** input exists, output path writable.  
- **Exception handling:** user-friendly dialogs on failures.  
- **Logs:** `%APPDATA%/AudioVerba/logs/YYYY-MM-DD.log`, rotated daily.

---

## 7. Development Roadmap

- **Phase 1: Prototype**  
  - PySide6 scaffold + drag-drop + FFmpeg conversion pipeline  
- **Phase 2: ASR**  
  - Integrate Reverb large model inference  
- **Phase 3: UI Polish & Config**  
  - Persist settings, implement multi-stage progress bars, logging  
- **Phase 4: Packaging & Testing**  
  - Windows installer, user acceptance testing, bug fixes

---

## 8. Next Steps

1. Kick off Phase 1: set up minimal PySide6 app and verify FFmpeg conversion.  
2. Acquire and validate Reverb large model weights on sample audio.  
3. Design initial config JSON schema and implement load/save routines.
