# Audio Transcription Tool

Fast, accurate audio transcription using faster-whisper. Generates timestamped SRT and VTT subtitle files with sentence-level precision.

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg

**Install FFmpeg:**
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

### Install Python Dependencies

```bash
pip install faster-whisper ffmpeg-python
```

Or using the included requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python transcribe.py input.mp3 output.srt
```

### Common Options

```bash
# High quality transcription
python transcribe.py input.wav output.srt --model medium --beam-size 8

# VTT format for web
python transcribe.py input.m4a output.vtt

# GPU acceleration
python transcribe.py input.mp3 output.srt --device cuda --compute-type float16

# Verbose logging
python transcribe.py input.mp3 output.srt --verbose
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | base | Model size: tiny, base, small, medium, large-v2, large-v3 |
| `--device` | cpu | Device: cpu, cuda, auto |
| `--compute-type` | int8 | Precision: int8, float16, float32 |
| `--language` | en | Language code or 'auto' for detection |
| `--beam-size` | 5 | Beam search width (1-10, higher is slower but more accurate) |
| `--no-vad` | - | Disable Voice Activity Detection |
| `--verbose` | - | Enable debug logging |

### Python API

```python
from transcribe import transcribe_to_srt, transcribe_to_vtt

# Basic transcription
transcribe_to_srt("lecture.mp3", "lecture.srt")

# Custom settings
transcribe_to_srt(
    audio_path="interview.wav",
    output_path="interview.srt",
    model_size="medium",
    beam_size=8
)

# VTT format
transcribe_to_vtt("podcast.m4a", "podcast.vtt")

# Batch processing
from pathlib import Path

for audio_file in Path("audio").glob("*.mp3"):
    output_file = audio_file.with_suffix(".srt")
    transcribe_to_srt(str(audio_file), str(output_file))
```

### VS Code Integration

VS Code tasks are pre-configured in `.vscode/tasks.json`:

1. Open an audio file in VS Code
2. Press `Ctrl+Shift+P` and select "Tasks: Run Task"
3. Choose a transcription task (base, small, medium, or custom)
4. Output file will be created in the same directory

## Model Selection

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | 40MB | Fastest | 85% | Quick drafts, testing |
| base | 75MB | Fast | 92% | General use (recommended) |
| small | 245MB | Moderate | 95% | Better quality |
| medium | 775MB | Slow | 97% | High quality needs |
| large-v2/v3 | 1.5GB | Slowest | 98% | Maximum accuracy |

**Processing time** (1 hour audio on Intel i7):
- tiny: 2-3 min
- base: 3-5 min
- small: 6-10 min
- medium: 12-20 min
- large-v3: 25-40 min

**GPU acceleration** (CUDA): 3-5x faster than CPU.

## Technical Details

### Architecture

The tool uses faster-whisper, a CTranslate2-optimized implementation of OpenAI's Whisper that provides 4x speed improvement with identical accuracy.

**Key components:**
- Whisper transformer model for speech recognition
- Silero VAD for silence detection and removal
- Beam search decoder for accuracy
- INT8/FP16 quantization for memory efficiency

**Timestamp accuracy:** Sentence-level with ~100ms precision. VAD filtering improves alignment by removing silence.

### Performance Characteristics

**Memory usage (base model):**
- CPU (int8): ~300MB
- GPU (float16): ~600MB

**Supported audio formats:** mp3, wav, m4a, flac, ogg, opus, webm

**Output formats:**
- SRT: Standard SubRip format with comma separators
- VTT: WebVTT format with period separators
- UTF-8 encoding for international characters

### Design Decisions

**faster-whisper over OpenAI Whisper:**
- 4x faster inference
- 70% lower memory usage (with int8 quantization)
- Identical accuracy
- Smaller dependency footprint

**Sentence-level over word-level timestamps:**
- Simpler implementation
- Faster processing
- Sufficient for most use cases (lectures, interviews, general transcription)
- Word-level alignment (WhisperX) adds complexity with minimal benefit for standard workflows

## Advanced Usage

### GPU Acceleration

CUDA support is included with faster-whisper. No additional installation required.

```bash
python transcribe.py audio.mp3 output.srt --device cuda --compute-type float16
```

Requirements: NVIDIA GPU with CUDA Compute Capability 7.0+

### Custom VAD Parameters

Edit the `transcribe.py` file to adjust silence detection:

```python
vad_parameters=dict(
    min_silence_duration_ms=1000,  # Default: 500
    speech_pad_ms=400               # Padding around speech
)
```

### Logging Configuration

The tool uses Python's standard logging module. Default level: INFO.

```bash
# Debug logging
python transcribe.py audio.mp3 output.srt --verbose
```

Logs include:
- Model loading status
- Language detection results
- Processing progress
- Output file location

## Troubleshooting

**"faster-whisper not found"**
```bash
pip install faster-whisper
```

**"FFmpeg error" or audio format issues**
```bash
# Verify FFmpeg is installed
ffmpeg -version

# Reinstall if needed (see Installation section)
```

**Slow processing**
```bash
# Use smaller model
python transcribe.py audio.mp3 output.srt --model tiny

# Or enable GPU
python transcribe.py audio.mp3 output.srt --device cuda
```

**Poor accuracy on technical content**
```bash
# Increase model size and beam width
python transcribe.py audio.mp3 output.srt --model medium --beam-size 10
```

**First run downloads model files**
- Model files (~75MB for base) download automatically
- Cached in `~/.cache/huggingface/`
- Subsequent runs are faster

## File Structure

```
.
├── transcribe.py          # Main script
├── requirements.txt       # Dependencies
├── README.md             # This file
├── example_usage.py      # API usage examples
├── setup.sh              # Automated setup script
└── .vscode/
    └── tasks.json        # VS Code task definitions
```

## Future Enhancements

**Word-level forced alignment:** Current implementation provides sentence-level timestamps. Word-level alignment (similar to WhisperX) would require integrating a phoneme-based forced alignment model. Implementation complexity: High. Use case: Language learning, karaoke-style subtitles.

**Speaker diarization:** Multi-speaker detection and labeling. Would require integrating pyannote.audio or similar. Implementation complexity: High. Use case: Interviews, meetings with multiple speakers.

## References

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - CTranslate2 optimized Whisper
- [OpenAI Whisper](https://github.com/openai/whisper) - Original model
- [CTranslate2](https://opennmt.net/CTranslate2/) - Inference optimization

## License

MIT
