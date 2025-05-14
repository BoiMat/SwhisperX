import warnings
warnings.filterwarnings("ignore")

import logging
logging.root.setLevel(logging.WARNING)
logging.getLogger("speechbrain.utils.quirks").setLevel(logging.ERROR)


import whisperx
import torch
import os
import csv
import argparse

from utils_git import format_time, save_transcription, whisperx_to_csv

HF_TOKEN = ''

def main():
    # Initialize the argument parser with a description.
    parser = argparse.ArgumentParser(description="Script for transcription and diarization")
    parser.add_argument('--audios_path', type=str, default='audio_files', help='Folder where the audio files are stored.')
    parser.add_argument('--transcriptions_path', type=str, default=None, help='Folder where the transcriptions will be stored.')
    parser.add_argument('--diarizations_path', type=str, default=None, help='Folder where the diarizations will be stored.')
    parser.add_argument('--whisper_model', type=str, default='small.en', help='Whisper model used under the hood.')
    parser.add_argument('--alignment_model', type=str, default=None, help='Alignment model used under the hood.')
    parser.add_argument('--language', type=str, default='de', help='The language code spoken in the audios')
    parser.add_argument('--min_speakers', type=int, default=None, help='Minimum number of speakers present in the audio.')
    parser.add_argument('--max_speakers', type=int, default=None, help='Maximum number of speakers present in the audio.')
    parser.add_argument('--hf_token', type=str, default=None, help='The HF token needed to access the speaker diarization model.')
    parser.add_argument('--diarize', action='store_true', help='Enable diarization mode')

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = "cuda"
    else:
        device = "cpu"
    print("Device:", device)
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    audio_dir = os.path.abspath(args.audios_path)
    parent_dir = os.path.dirname(audio_dir)

    files = os.listdir(audio_dir)
    audio_files = [os.path.join(audio_dir, f) for f in files if os.path.isfile(os.path.join(audio_dir, f))]

    if args.diarize and args.hf_token is None:
        print("WARNING: The HuggingFace token is required for the speaker diarization but it's not provided. Diarization skipped.")
        args.diarize = False

    if args.transcriptions_path is None:
        default_trans_dir = os.path.join(parent_dir, "transcriptions")
        os.makedirs(default_trans_dir, exist_ok=True)
        args.transcriptions_path = default_trans_dir

    if args.diarize and args.diarizations_path is None:
        default_diari_dir = os.path.join(parent_dir, "diarizations")
        os.makedirs(default_diari_dir, exist_ok=True)
        args.diarizations_path = default_diari_dir

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(args.whisper_model, device=device, compute_type=compute_type, language=args.language) # or "small.en"

    for i, audio_file in enumerate(audio_files):

        print(f"\nProcessing audio {i+1}/{len(audio_files)}: {audio_file}")

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        save_transcription(result, audio_file, output_dir=args.transcriptions_path)

        if args.diarize:
            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            # 3. Assign speaker labels
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

            # add min/max number of speakers if known
            diarize_segments = diarize_model(audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers)

            result = whisperx.assign_word_speakers(diarize_segments, result)

            whisperx_to_csv(result, audio_file, output_dir=args.diarizations_path)

if __name__ == "__main__":
    main()