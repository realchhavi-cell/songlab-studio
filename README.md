# SongLab

Your Personal Music Studio - Karaoke Maker, Stem Separator & Transposer

## Features

- **Karaoke Maker** - Remove vocals from any song to create instrumental tracks
- **YouTube Download** - Download audio directly from YouTube
- **Stem Separator** - Split songs into drums, bass, vocals, and other instruments
- **Transpose** - Change the key/pitch of any audio file

## Installation

### Prerequisites
- Python 3.10+
- FFmpeg
- yt-dlp

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/songlab.git
cd songlab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or run the setup script
bash setup.sh
```

## Usage

### Web UI (Recommended)

```bash
./songlab
```

Then open http://127.0.0.1:7860 in your browser.

### Command Line

```bash
# Create karaoke track from YouTube
./songlab karaoke "https://youtube.com/watch?v=..."

# Remove vocals from local file
./songlab vocals song.mp3

# Extract only vocals
./songlab isolate song.mp3

# Separate into all 4 stems
./songlab stems song.mp3

# Transpose audio
./songlab transpose song.mp3 -s -2  # Down 2 semitones
./songlab transpose song.mp3 -s 3   # Up 3 semitones

# Just download from YouTube
./songlab download "https://youtube.com/watch?v=..."
```

## How It Works

SongLab uses [Demucs](https://github.com/facebookresearch/demucs) by Meta AI for high-quality audio source separation. The AI model separates audio into four stems:
- Drums
- Bass
- Vocals
- Other (guitars, synths, etc.)

For karaoke tracks, it combines everything except vocals into an instrumental mix.

## Tech Stack

- **Demucs** - AI-powered audio source separation
- **PyTorch** - Deep learning framework
- **Gradio** - Web UI framework
- **FFmpeg** - Audio processing
- **yt-dlp** - YouTube downloading

## License

MIT
