#!/usr/bin/env python3
"""
Audio transcription tool using faster-whisper.
Converts audio files to SRT/VTT subtitle formats with accurate timestamps.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import timedelta

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("ERROR: faster-whisper is not installed.")
    print("Install with: pip install faster-whisper")
    sys.exit(1)

try:
    import ffmpeg
except ImportError:
    print("WARNING: ffmpeg-python is not installed.")
    print("Install with: pip install ffmpeg-python")
    print("Note: FFmpeg binaries must also be installed on your system")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    pass


def format_timestamp_srt(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm)

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    millis = int((td.total_seconds() % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    Convert seconds to VTT timestamp format (HH:MM:SS.mmm)

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    return format_timestamp_srt(seconds).replace(',', '.')


def validate_audio_file(audio_path: str) -> Path:
    """
    Validate audio file existence and format support.

    Args:
        audio_path: Path to audio file

    Returns:
        Path object of the audio file

    Raises:
        TranscriptionError: If file doesn't exist or format is unsupported
    """
    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm'}
    if audio_file.suffix.lower() not in supported_formats:
        raise TranscriptionError(
            f"Unsupported audio format: {audio_file.suffix}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    return audio_file


def transcribe_to_srt(
    audio_path: str,
    output_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    beam_size: int = 5,
    vad_filter: bool = True
) -> None:
    """
    Transcribe audio file to SRT subtitle format.

    Args:
        audio_path: Path to input audio file
        output_path: Path to output SRT file
        model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
        device: Device to run on ('cpu', 'cuda', or 'auto')
        compute_type: Compute type for faster-whisper (int8, float16, float32)
        language: Language code (default: 'en' for English)
        beam_size: Beam size for decoding
        vad_filter: Enable Voice Activity Detection to filter silence

    Raises:
        TranscriptionError: If transcription fails
    """
    audio_file = validate_audio_file(audio_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Whisper model: {model_size}")
    logger.info(f"Device: {device}, Compute Type: {compute_type}")

    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    except Exception as e:
        raise TranscriptionError(f"Failed to load Whisper model: {e}")

    logger.info(f"Transcribing: {audio_file.name}")

    try:
        segments, info = model.transcribe(
            str(audio_file),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False
        )

        logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2%})")

        with open(output_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, start=1):
                f.write(f"{i}\n")

                start_time = format_timestamp_srt(segment.start)
                end_time = format_timestamp_srt(segment.end)
                f.write(f"{start_time} --> {end_time}\n")

                f.write(f"{segment.text.strip()}\n")
                f.write("\n")

                if i % 10 == 0:
                    logger.debug(f"Processed {i} segments")

        logger.info(f"Transcription complete. Output saved to: {output_file}")
        logger.info(f"Total segments: {i}")

    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")


def transcribe_to_vtt(
    audio_path: str,
    output_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    beam_size: int = 5,
    vad_filter: bool = True
) -> None:
    """
    Transcribe audio file to VTT subtitle format.

    Args: Same as transcribe_to_srt

    Raises:
        TranscriptionError: If transcription fails
    """
    audio_file = validate_audio_file(audio_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Whisper model: {model_size}")
    logger.info(f"Device: {device}, Compute Type: {compute_type}")

    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    except Exception as e:
        raise TranscriptionError(f"Failed to load Whisper model: {e}")

    logger.info(f"Transcribing: {audio_file.name}")

    try:
        segments, info = model.transcribe(
            str(audio_file),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False
        )

        logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2%})")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")

            for i, segment in enumerate(segments, start=1):
                f.write(f"{i}\n")

                start_time = format_timestamp_vtt(segment.start)
                end_time = format_timestamp_vtt(segment.end)
                f.write(f"{start_time} --> {end_time}\n")

                f.write(f"{segment.text.strip()}\n")
                f.write("\n")

                if i % 10 == 0:
                    logger.debug(f"Processed {i} segments")

        logger.info(f"Transcription complete. Output saved to: {output_file}")
        logger.info(f"Total segments: {i}")

    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to SRT/VTT subtitles using faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py input.mp3 output.srt
  python transcribe.py input.wav output.vtt --model medium
  python transcribe.py input.m4a output.srt --device cuda --compute-type float16
  python transcribe.py input.mp3 output.srt --model large-v3 --beam-size 10

Model Sizes (speed vs accuracy):
  tiny, base   - Fastest, good for quick drafts
  small        - Balanced speed and accuracy
  medium       - Better accuracy, slower
  large-v2/v3  - Best accuracy, slowest
        """
    )

    parser.add_argument(
        "audio_file",
        help="Path to input audio file (.mp3, .wav, .m4a, etc.)"
    )

    parser.add_argument(
        "output_file",
        help="Path to output subtitle file (.srt or .vtt)"
    )

    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: base)"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to run on (default: cpu)"
    )

    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type for model (default: int8 for CPU efficiency)"
    )

    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en). Use 'auto' for auto-detection"
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5, higher = more accurate but slower)"
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    output_ext = Path(args.output_file).suffix.lower()

    if output_ext not in ['.srt', '.vtt']:
        logger.error(f"Unsupported output format: {output_ext}")
        logger.error("Supported formats: .srt, .vtt")
        sys.exit(1)

    language = None if args.language == 'auto' else args.language

    try:
        if output_ext == '.srt':
            transcribe_to_srt(
                audio_path=args.audio_file,
                output_path=args.output_file,
                model_size=args.model,
                device=args.device,
                compute_type=args.compute_type,
                language=language,
                beam_size=args.beam_size,
                vad_filter=not args.no_vad
            )
        else:
            transcribe_to_vtt(
                audio_path=args.audio_file,
                output_path=args.output_file,
                model_size=args.model,
                device=args.device,
                compute_type=args.compute_type,
                language=language,
                beam_size=args.beam_size,
                vad_filter=not args.no_vad
            )

    except TranscriptionError as e:
        logger.error(f"Transcription error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Transcription cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
