---
title: SongLab Studio
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# SongLab

Your Personal Music Studio - Karaoke Maker, Stem Separator & Transposer

## Features

- **Karaoke Maker** - Remove vocals from any song to create instrumental tracks
- **YouTube Download** - Download audio directly from YouTube
- **Stem Separator** - Split songs into drums, bass, vocals, and other instruments
- **Transpose** - Change the key/pitch of any audio file

## Usage

### Web UI

Use the tabs to:
1. **Karaoke Maker** - Paste YouTube URL or upload a file to get instrumental + vocals
2. **YouTube Download** - Download audio from YouTube
3. **Stem Separator** - Split into drums, bass, vocals, other
4. **Transpose** - Change key with slider

## Tech Stack

- **Demucs** - AI-powered audio source separation by Meta
- **PyTorch** - Deep learning framework
- **Gradio** - Web UI framework
- **FFmpeg** - Audio processing
- **yt-dlp** - YouTube downloading
