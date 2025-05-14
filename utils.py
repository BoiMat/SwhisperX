import whisperx
import os
import csv

def format_time(seconds):
    # Calculate hours, minutes, and seconds (including milliseconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60  # This includes the fractional part
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def save_transcription(result, audio_file_path, output_dir="transcriptions"):
    """
    Save the transcription result to a CSV file using tab as delimiter.

    Parameters:
        result (dict): The transcription result from whisperx containing "segments".
        audio_file_path (str): Path to the original audio file, used to name the output file.
        output_dir (str): Directory where the output file will be saved. Defaults to "transcriptions".
    """
    # Create output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the base name of the audio file and form the output file path.
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.csv")
    
    # Open the CSV file for writing with tab delimiter.
    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        # Write the header row.
        writer.writerow(["Start", "End", "Text"])
        
        # Write each segment as a row.
        for segment in result.get("segments", []):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "")
            writer.writerow([format_time(start_time), format_time(end_time), text])
    
    print(f"Transcription saved to {output_file}\n")

def whisperx_to_csv(result, audio_file, output_dir="diarizations"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate output filename by replacing the audio extension with .tsv
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.csv")

    rows = []

    for segment in result.get('segments', []):
        words = segment.get('words', [])
        if not words:
            continue

        # Get the speaker for the first word, defaulting to "UNKNOWN" if missing
        first_speaker = words[0].get('speaker') if words[0].get('speaker') not in (None, '') else "UNKNOWN"
        current_speaker = first_speaker
        current_text = words[0]['word']
        start_time = words[0]['start']
        end_time = words[0]['end']
        
        for word in words[1:]:
            # Use "UNKNOWN" if the speaker field is missing or empty
            speaker = word.get('speaker') if word.get('speaker') not in (None, '') else "UNKNOWN"
            if speaker == current_speaker:
                # Continue with the current grouping if the speaker is the same
                current_text += " " + word['word']
                end_time = word['end']
            else:
                # Save the current group with formatted times
                rows.append([format_time(start_time), format_time(end_time), current_speaker, current_text])
                current_speaker = speaker
                current_text = word['word']
                start_time = word['start']
                end_time = word['end']
        
        # Append the last group for this segment
        rows.append([format_time(start_time), format_time(end_time), current_speaker, current_text])

    # Write the results to a tab-delimited file (TSV)
    with open(output_file, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Start', 'End', 'Speaker', 'Text'])
        writer.writerows(rows)

    print(f"Diarization file saved to: {output_file}\n")