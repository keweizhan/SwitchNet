#!/usr/bin/env python3
"""
Example usage patterns for the transcription module.
Demonstrates various API usage scenarios.
"""

from transcribe import transcribe_to_srt, transcribe_to_vtt, TranscriptionError
from pathlib import Path


def example_basic():
    """Basic SRT transcription"""
    print("Example 1: Basic SRT transcription")
    print("-" * 50)

    try:
        transcribe_to_srt(
            audio_path="sample.mp3",
            output_path="output.srt"
        )
        print("Success: Transcription complete")
    except TranscriptionError as e:
        print(f"Error: {e}")


def example_high_quality():
    """High-quality transcription with custom settings"""
    print("\nExample 2: High-quality transcription")
    print("-" * 50)

    try:
        transcribe_to_srt(
            audio_path="interview.wav",
            output_path="interview_hq.srt",
            model_size="medium",
            beam_size=8,
            vad_filter=True
        )
        print("Success: Transcription complete")
    except TranscriptionError as e:
        print(f"Error: {e}")


def example_vtt_format():
    """VTT format for web videos"""
    print("\nExample 3: VTT format")
    print("-" * 50)

    try:
        transcribe_to_vtt(
            audio_path="podcast.m4a",
            output_path="podcast.vtt",
            model_size="small"
        )
        print("Success: Transcription complete")
    except TranscriptionError as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Batch processing multiple files"""
    print("\nExample 4: Batch processing")
    print("-" * 50)

    audio_files = [
        "lecture_01.mp3",
        "lecture_02.mp3",
        "lecture_03.mp3"
    ]

    for audio_file in audio_files:
        audio_path = Path(audio_file)

        if not audio_path.exists():
            print(f"Skipping {audio_file} (not found)")
            continue

        output_path = audio_path.with_suffix('.srt')

        try:
            print(f"Processing {audio_file}...")
            transcribe_to_srt(
                audio_path=str(audio_path),
                output_path=str(output_path),
                model_size="base"
            )
            print(f"Complete: {audio_file} -> {output_path}")
        except TranscriptionError as e:
            print(f"Failed: {audio_file} - {e}")


def example_gpu_acceleration():
    """GPU-accelerated transcription (requires CUDA)"""
    print("\nExample 5: GPU acceleration")
    print("-" * 50)

    try:
        transcribe_to_srt(
            audio_path="long_audio.wav",
            output_path="long_audio.srt",
            model_size="base",
            device="cuda",
            compute_type="float16"
        )
        print("Success: Transcription complete")
    except TranscriptionError as e:
        print(f"Error: {e}")
        print("Note: GPU acceleration requires CUDA-enabled GPU")


def example_auto_language_detection():
    """Auto-detect language"""
    print("\nExample 6: Auto language detection")
    print("-" * 50)

    try:
        transcribe_to_srt(
            audio_path="multilingual.mp3",
            output_path="multilingual.srt",
            language=None,
            model_size="small"
        )
        print("Success: Transcription complete")
    except TranscriptionError as e:
        print(f"Error: {e}")


def example_custom_output_location():
    """Custom output directory"""
    print("\nExample 7: Custom output location")
    print("-" * 50)

    try:
        output_dir = Path("transcripts")
        output_dir.mkdir(exist_ok=True)

        transcribe_to_srt(
            audio_path="meeting.mp3",
            output_path="transcripts/meeting_2024_02_14.srt",
            model_size="base"
        )
        print("Success: Transcription complete")
    except TranscriptionError as e:
        print(f"Error: {e}")


def example_speed_vs_quality():
    """Speed vs quality comparison"""
    print("\nExample 8: Speed vs Quality comparison")
    print("-" * 50)

    configs = [
        ("Fast (draft)", "tiny", 1),
        ("Balanced", "base", 5),
        ("High Quality", "medium", 8)
    ]

    for name, model, beam_size in configs:
        try:
            print(f"\nProcessing with {name} settings...")
            transcribe_to_srt(
                audio_path="sample.mp3",
                output_path=f"output_{model}.srt",
                model_size=model,
                beam_size=beam_size
            )
            print(f"Complete: {name}")
        except TranscriptionError as e:
            print(f"Failed: {name} - {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Transcription API Examples")
    print("=" * 50)

    # Uncomment the examples you want to run:

    # example_basic()
    # example_high_quality()
    # example_vtt_format()
    # example_batch_processing()
    # example_gpu_acceleration()
    # example_auto_language_detection()
    # example_custom_output_location()
    # example_speed_vs_quality()

    print("\n" + "=" * 50)
    print("Note: Uncomment examples in __main__ to run them")
    print("Edit the audio file paths to match your files")
    print("=" * 50)
