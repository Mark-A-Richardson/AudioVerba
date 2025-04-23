# Setting Up Reverb ASR with GPU Support

Below are the steps your programmer needs to follow to pull the Reverb source code, install dependencies, and run inference on GPU.

---

## 1. Clone the Reverb Repository

1. Open a terminal on your development machine.
2. Create and navigate to your projects directory:
   ```bash
   mkdir -p ~/projects/reverb
   cd ~/projects/reverb
   ```
3. Clone the official Reverb ASR repository and enter its directory:
   ```bash
   git clone https://github.com/revdotcom/reverb.git
   cd reverb
   ```

---

## 2. Install Python Dependencies

Use a virtual environment and install requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Then install CUDA‑enabled PyTorch wheels:

```bash
pip install torch==2.2.2+cu121 \
            torchvision==0.17.2+cu121 \
            torchaudio==2.2.2+cu121 \
            --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Prepare Local Model Files

Reverb’s Python API expects you to supply model configuration and checkpoint files locally:

1. Download the `config.yaml` and `.pt` checkpoint for `reverb_asr_v1`.
2. Organize them under a `models` folder, for example:
   ```
   reverb/
   └── models/
       └── reverb_asr_v1/
           ├── config.yaml
           └── reverb_asr_v1.pt
   ```

---

## 4. Instantiate the ASR Engine with GPU Enabled

Bypass the zero‑argument helper and create the `ReverbASR` instance directly with a GPU index:

```python
from wenet.cli.reverb import ReverbASR

# Paths to your local model files\config_path     = "models/reverb_asr_v1/config.yaml"
checkpoint_path = "models/reverb_asr_v1/reverb_asr_v1.pt"

# Instantiate with GPU index 0 (enables CUDA)
reverb = ReverbASR(
    config_path,
    checkpoint_path,
    gpu=0,
    overwrite_cmvn=False
)

# Run transcription on GPU
audio_path = "path/to/audio.mp3"
output = reverb.transcribe(audio_path)
print(output)
```

---

## 5. Verify GPU Usage

Ensure CUDA is active before transcription:

```bash
python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
EOF
```

If `CUDA available: True` and a valid device index prints, your setup is ready.

---

### Summary

- **Clone** the `reverb` repo.
- **Install** dependencies and prebuilt CUDA wheels.
- **Place** model files in a `models/reverb_asr_v1/` directory.
- **Instantiate** `ReverbASR` with `gpu=0`.
- **Verify** CUDA availability.

Your programmer can now run Reverb ASR on GPU for efficient transcription.