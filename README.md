# SwhisperX

### WhisperX Fine-Tuning & Transcription Toolkit for Swiss German.

This repository provides:

- **`WhisperX.py`**: Script to transcribe (and optionally diarize) audio using WhisperX.
- **`finetune.py`**: Script to fine-tune a Whisper model on your data.

You can run everything **inside the whisperx container** (if available) or directly in your Python environment.

---

## 1. Prerequisites

- **Container**: If you have access to the `whisperx.sif`, prefix commands with:
  ```bash
  apptainer exec --nv <path-to-the-container>/whisperx.sif <your command>
  ```

* **Huggingface Token**:\
  For speaker diarization you **must** supply an HF token with access to the [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) model.

---

## 2. Transcription & Diarization with WhisperX

Use `WhisperX.py` to do VAD, transcription, forced alignment, and optional speaker diarization.

### 2.1 Basic Transcription

```bash
python3 WhisperX_git.py \
    --audios_path ./my_audio \
    --transcriptions_path ./my_transcripts \
    --whisper_model openai/whisper-large
```

### 2.2 Transcription + Diarization

```bash
apptainer exec --nv whisperx.sif \
  python3 WhisperX_git.py \
    --audios_path ./my_audio \
    --transcriptions_path ./my_transcripts \
    --diarizations_path ./my_diarizations \
    --whisper_model openai/whisper-large \
    --hf_token $HF_TOKEN \
    --diarize \
```

* `--hf_token` must be a token **with access** to the Pyannote speaker-diarization-3.1 model.
* Adjust `--min_speakers`/`--max_speakers` to your data.

---
### 2.3 Command-Line Arguments
Here are the available command-line options and what they do:

* **--audios\_path**
  Folder where your input audio files live (default: `audio_files`).

* **--transcriptions\_path**
  (Optional) Folder to which the generated transcription CSVs are written (if not set, the default location is the "transcriptions" folder in the same parent directory as the audio folders).

* **--diarizations\_path**
  (Optional) Folder to which the speaker-diarization outputs are written (only if you enable diarization, if not set, the default location is the "diarizations" folder in the same parent directory as the audio folders).

* **--whisper\_model**
  Which Whisper checkpoint to use for transcription (e.g., `small.en`, `openai/whisper-large`).

* **--alignment\_model**
  (Optional) Forced-alignment model to use for precise word-level timestamps.

* **--language**
  ISO language code of the audio (default: `de` for German). It automatically sets the alignment model based on WhisperX default choice for the specified language. 

* **--min\_speakers**
  (Optional) Minimum number of speakers to detect when diarizing.

* **--max\_speakers**
  (Optional) Maximum number of speakers to detect when diarizing.

* **--hf\_token**
  Your Hugging Face access token (required if you use diarization).

* **--diarize**
  Flag to turn on speaker diarization (requires `--hf_token` and `pyannote/speaker-diarization-3.1`).


