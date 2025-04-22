# AudioVerba

Desktop audio transcription tool powered by Reverb ASR.

## Project Setup

1. Install Poetry: `pip install poetry`
2. Install dependencies: `poetry install`

### Hugging Face Setup (Required for Transcription and Diarization)

AudioVerba uses models hosted on Hugging Face Hub. Some of these require you to accept their terms of use and provide an authentication token.

1.  **Create a Hugging Face Account:** If you don't have one, sign up at [https://huggingface.co/join](https://huggingface.co/join).
2.  **Accept Model Terms:** Log in to your Hugging Face account and visit the following pages. Accept the terms and conditions presented on each page:
    *   **Reverb ASR:** [https://huggingface.co/Revai/reverb-asr](https://huggingface.co/Revai/reverb-asr)
    *   **Speaker Diarization Pipeline:** [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    *   **Segmentation Model (used by diarization):** [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
    *   *(Note: The diarization pipeline might also implicitly use an embedding model like `pyannote/embedding`. You might need to accept its terms too if you encounter further issues.)*
3.  **Generate an Access Token:** Go to your Hugging Face account settings: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Create a new access token with at least 'read' permissions. Copy the generated token.
4.  **Create `.env` File:** In the root directory of the AudioVerba project, create a file named `.env`.
5.  **Add Token to `.env`:** Add the following line to your `.env` file, replacing `hf_YOUR_TOKEN_HERE` with the actual token you copied:
    ```
    HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"
    ```
6.  **(Optional) Suppress Symlink Warning on Windows:** To hide warnings about file caching on Windows, you can add this line to your `.env` file:
    ```
    HF_HUB_DISABLE_SYMLINKS_WARNING=1
    ```

## Running the Application

`poetry run python src/audioverba/main.py`
