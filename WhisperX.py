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

from utils import format_time, save_transcription, whisperx_to_csv, save_word_timestamps

def main():
    # Initialize the argument parser with a description.
    parser = argparse.ArgumentParser(description="Script for transcription and diarization")
    parser.add_argument('--audios_path', type=str, default='audio_files', help='Folder where the audio files are stored.')
    parser.add_argument('--transcriptions_path', type=str, default=None, help='Folder where the transcriptions will be stored.')
    parser.add_argument('--word_timestamps_path', type=str, default=None, help='Where to save the word-timestamp TSVs (if --word_timestamps).')
    parser.add_argument('--diarizations_path', type=str, default=None, help='Folder where the diarizations will be stored.')
    parser.add_argument('--whisper_model', type=str, default='small.en', help='Whisper model used under the hood.')
    parser.add_argument('--alignment_model', type=str, default=None, help='Alignment model used under the hood.')
    parser.add_argument('--language', type=str, default='de', help='The language code spoken in the audios')
    parser.add_argument('--min_speakers', type=int, default=None, help='Minimum number of speakers present in the audio.')
    parser.add_argument('--max_speakers', type=int, default=None, help='Maximum number of speakers present in the audio.')
    parser.add_argument('--hf_token', type=str, default=None, help='The HF token needed to access the speaker diarization model.')
    parser.add_argument('--word_timestamps', action='store_true', help='Run alignment and save per-word timestamps.')
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

    audio_path = os.path.abspath(args.audios_path)
    parent_dir = os.path.dirname(audio_path)
    if os.path.isdir(audio_path):
        files = os.listdir(audio_path)
        audio_files = [os.path.join(audio_path, f) for f in files if os.path.isfile(os.path.join(audio_path, f))]
    elif os.path.isfile(audio_path):
        # it's a single file → just process that one
        audio_files = [audio_path]
    else:
        print(f"ERROR: `{audio_path}` is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    if args.diarize and not args.hf_token:
        print(
            "WARNING: HuggingFace token required for diarization not provided – skipping diarization."
        )
        args.diarize = False

    if args.transcriptions_path is None:
        default_trans_dir = os.path.join(parent_dir, "transcriptions")
        os.makedirs(default_trans_dir, exist_ok=True)
        args.transcriptions_path = default_trans_dir
    
    if args.word_timestamps and args.word_timestamps_path is None:
        default_wlt_dir = os.path.join(parent_dir, "word_timestamps")
        os.makedirs(default_wlt_dir, exist_ok=True)
        args.word_timestamps_path = default_wlt_dir

    if args.diarize and args.diarizations_path is None:
        default_diari_dir = os.path.join(parent_dir, "diarizations")
        os.makedirs(default_diari_dir, exist_ok=True)
        args.diarizations_path = default_diari_dir

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(args.whisper_model, device=device, compute_type=compute_type, language=args.language)

    for i, audio_file in enumerate(audio_files, start=1):
        print(f"\nProcessing {i}/{len(audio_files)}: {audio_file!r}")

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        save_transcription(result, audio_file, output_dir=args.transcriptions_path)

        if args.word_timestamps or args.diarize:

            model_a, metadata = whisperx.load_align_model(language_code=args.language, device=device)
            aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            if args.word_timestamps:
                save_word_timestamps(aligned, audio_file, output_dir=args.word_timestamps_path)

        if args.diarize:

            diarize_model = whisperx.DiarizationPipeline(use_auth_token=args.hf_token, device=device)
            diarize_segments = diarize_model(audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers)
            diarized = whisperx.assign_word_speakers(diarize_segments, aligned)

            whisperx_to_csv(diarized, audio_file, output_dir=args.diarizations_path)

if __name__ == "__main__":
    main()