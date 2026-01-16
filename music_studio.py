#!/usr/bin/env python3
"""
Music Studio - Your personal karaoke and music production toolkit

Features:
  - Download songs from YouTube
  - Remove vocals (create instrumentals for karaoke)
  - Extract individual stems (vocals, drums, bass, other)
  - Transpose (pitch shift) audio
  - Combine operations in one command

Usage:
    python music_studio.py karaoke "youtube_url"           # Download & remove vocals
    python music_studio.py vocals song.mp3                  # Remove vocals from file
    python music_studio.py stems song.mp3                   # Extract all stems
    python music_studio.py transpose song.mp3 -s 3          # Pitch shift
    python music_studio.py download "youtube_url"           # Just download
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path
import tempfile

# Project directory - all outputs go here
PROJECT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_DIR / "output"

# Demucs model options (htdemucs is the best quality)
DEMUCS_MODEL = "htdemucs"


def print_banner():
    """Print a nice banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                      MUSIC STUDIO                            ║
    ║         Karaoke Maker | Stem Separator | Transposer          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies(need_demucs=False):
    """Check if required tools are installed."""
    missing = []

    if not shutil.which('ffmpeg'):
        missing.append('ffmpeg')

    if not shutil.which('yt-dlp') and not shutil.which('youtube-dl'):
        missing.append('yt-dlp')

    if need_demucs:
        try:
            import demucs
        except ImportError:
            missing.append('demucs (pip install demucs)')

    if missing:
        print("Missing dependencies:", ', '.join(missing))
        print("\nRun the setup script to install them:")
        print(f"  bash {PROJECT_DIR}/setup.sh")
        print("\nOr install manually:")
        print("  brew install ffmpeg yt-dlp")
        print("  pip install demucs")
        sys.exit(1)


def get_ytdlp():
    """Get the YouTube downloader command."""
    if shutil.which('yt-dlp'):
        return 'yt-dlp'
    elif shutil.which('youtube-dl'):
        return 'youtube-dl'
    return None


def download_from_youtube(url, output_dir=None):
    """Download audio from YouTube URL."""
    ytdlp = get_ytdlp()
    if not ytdlp:
        print("Error: yt-dlp or youtube-dl not found")
        sys.exit(1)

    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, '%(title)s.%(ext)s')

    print(f"\nDownloading from YouTube: {url}")

    cmd = [
        ytdlp,
        '-x',  # Extract audio
        '--audio-format', 'mp3',
        '--audio-quality', '0',  # Best quality
        '-o', output_template,
        '--print', 'after_move:filepath',
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        sys.exit(1)

    downloaded_file = result.stdout.strip().split('\n')[-1]
    print(f"Downloaded: {downloaded_file}")
    return downloaded_file


def separate_stems(input_file, output_dir=None, model=DEMUCS_MODEL, stems_to_keep=None):
    """
    Separate audio into stems using Demucs.

    Args:
        input_file: Path to input audio file
        output_dir: Output directory (default: output/stems/)
        model: Demucs model to use (htdemucs, htdemucs_ft, mdx_extra)
        stems_to_keep: List of stems to keep (default: all - vocals, drums, bass, other)

    Returns:
        Dictionary mapping stem names to file paths
    """
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torchaudio

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "stems"
    os.makedirs(output_dir, exist_ok=True)

    song_name = input_path.stem
    song_output_dir = output_dir / song_name
    os.makedirs(song_output_dir, exist_ok=True)

    print(f"\nSeparating stems from: {input_file}")
    print(f"Using model: {model}")
    print("This may take a few minutes depending on song length and your hardware...")

    # Load the model
    print("\nLoading Demucs model...")
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_obj = get_model(model)
    model_obj.to(device)
    model_obj.eval()

    # Load audio
    print("Loading audio file...")
    wav, sr = torchaudio.load(input_file)

    # Resample if necessary (Demucs expects 44100 Hz)
    if sr != model_obj.samplerate:
        print(f"Resampling from {sr} Hz to {model_obj.samplerate} Hz...")
        resampler = torchaudio.transforms.Resample(sr, model_obj.samplerate)
        wav = resampler(wav)
        sr = model_obj.samplerate

    # Convert to stereo if mono
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]  # Take first 2 channels

    # Add batch dimension and move to device
    wav = wav.unsqueeze(0).to(device)

    # Apply model
    print("Separating stems (this is the slow part)...")
    with torch.no_grad():
        sources = apply_model(model_obj, wav, device=device, progress=True)

    # Save stems
    sources = sources.squeeze(0).cpu()
    stem_names = model_obj.sources  # Usually: ['drums', 'bass', 'other', 'vocals']

    stem_files = {}
    for i, stem_name in enumerate(stem_names):
        if stems_to_keep and stem_name not in stems_to_keep:
            continue

        stem_path = song_output_dir / f"{stem_name}.mp3"
        stem_wav = sources[i]

        # Save as mp3 via temp wav
        temp_wav = song_output_dir / f"{stem_name}_temp.wav"
        torchaudio.save(str(temp_wav), stem_wav, sr)

        # Convert to mp3
        subprocess.run([
            'ffmpeg', '-y', '-i', str(temp_wav),
            '-acodec', 'libmp3lame', '-q:a', '2',
            str(stem_path)
        ], capture_output=True)

        temp_wav.unlink()  # Remove temp file

        stem_files[stem_name] = str(stem_path)
        print(f"  Saved: {stem_path}")

    return stem_files


def remove_vocals(input_file, output_file=None):
    """
    Remove vocals from audio to create an instrumental/karaoke version.

    This combines all non-vocal stems (drums, bass, other) into one file.
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    if not output_file:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = OUTPUT_DIR / f"{input_path.stem}_instrumental.mp3"

    output_path = Path(output_file)

    print(f"\nRemoving vocals from: {input_file}")
    print("Creating instrumental/karaoke version...")
    print("This may take a few minutes...")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = get_model(DEMUCS_MODEL)
    model.to(device)
    model.eval()

    # Load audio
    print("Loading audio...")
    wav, sr = torchaudio.load(input_file)

    if sr != model.samplerate:
        resampler = torchaudio.transforms.Resample(sr, model.samplerate)
        wav = resampler(wav)
        sr = model.samplerate

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    wav = wav.unsqueeze(0).to(device)

    # Separate
    print("Separating stems...")
    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=True)

    sources = sources.squeeze(0).cpu()
    stem_names = model.sources  # ['drums', 'bass', 'other', 'vocals']

    # Combine everything except vocals
    print("Creating instrumental mix...")
    instrumental = None
    for i, stem_name in enumerate(stem_names):
        if stem_name != 'vocals':
            if instrumental is None:
                instrumental = sources[i]
            else:
                instrumental = instrumental + sources[i]

    # Save instrumental
    temp_wav = output_path.with_suffix('.wav')
    torchaudio.save(str(temp_wav), instrumental, sr)

    # Convert to mp3
    subprocess.run([
        'ffmpeg', '-y', '-i', str(temp_wav),
        '-acodec', 'libmp3lame', '-q:a', '2',
        str(output_path)
    ], capture_output=True)

    temp_wav.unlink()

    print(f"\nInstrumental saved to: {output_path}")
    return str(output_path)


def extract_vocals_only(input_file, output_file=None):
    """
    Extract only the vocals from a song.
    Useful for acapella or remixing.
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    if not output_file:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = OUTPUT_DIR / f"{input_path.stem}_vocals.mp3"

    output_path = Path(output_file)

    print(f"\nExtracting vocals from: {input_file}")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = get_model(DEMUCS_MODEL)
    model.to(device)
    model.eval()

    # Load audio
    wav, sr = torchaudio.load(input_file)

    if sr != model.samplerate:
        resampler = torchaudio.transforms.Resample(sr, model.samplerate)
        wav = resampler(wav)
        sr = model.samplerate

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    wav = wav.unsqueeze(0).to(device)

    # Separate
    print("Processing...")
    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=True)

    sources = sources.squeeze(0).cpu()
    stem_names = model.sources

    # Get vocals
    vocals_idx = stem_names.index('vocals')
    vocals = sources[vocals_idx]

    # Save vocals
    temp_wav = output_path.with_suffix('.wav')
    torchaudio.save(str(temp_wav), vocals, sr)

    subprocess.run([
        'ffmpeg', '-y', '-i', str(temp_wav),
        '-acodec', 'libmp3lame', '-q:a', '2',
        str(output_path)
    ], capture_output=True)

    temp_wav.unlink()

    print(f"\nVocals saved to: {output_path}")
    return str(output_path)


def transpose_audio(input_file, semitones, output_file=None, speed_compensation=True):
    """
    Transpose audio by the specified number of semitones.
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    if not output_file:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sign = "+" if semitones >= 0 else ""
        st_str = f"{semitones:.1f}".rstrip('0').rstrip('.')
        suffix = f"_transposed_{sign}{st_str}st"
        output_file = OUTPUT_DIR / f"{input_path.stem}{suffix}{input_path.suffix}"

    output_path = Path(output_file)

    pitch_multiplier = 2 ** (semitones / 12)

    print(f"\nTransposing: {input_file}")
    print(f"Shift: {semitones:+.1f} semitones")

    is_video = input_path.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']

    # Get sample rate
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
                 str(input_path)]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    sample_rate = int(probe_result.stdout.strip()) if probe_result.stdout.strip() else 44100

    if speed_compensation:
        new_rate = int(sample_rate * pitch_multiplier)
        tempo_factor = 1 / pitch_multiplier

        atempo_filters = []
        remaining = tempo_factor
        while remaining > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining /= 0.5
        atempo_filters.append(f"atempo={remaining:.6f}")

        filter_complex = f"asetrate={new_rate},aresample={sample_rate}," + ",".join(atempo_filters)
    else:
        new_rate = int(sample_rate * pitch_multiplier)
        filter_complex = f"asetrate={new_rate},aresample={sample_rate}"

    if is_video:
        cmd = ['ffmpeg', '-y', '-i', str(input_path), '-af', filter_complex, '-c:v', 'copy', str(output_path)]
    else:
        cmd = ['ffmpeg', '-y', '-i', str(input_path), '-af', filter_complex, str(output_path)]

    print("Processing...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(f"Transposed file saved to: {output_path}")
    return str(output_path)


def karaoke_mode(source, semitones=0, output_file=None):
    """
    Full karaoke workflow: Download (if URL) -> Remove vocals -> Optionally transpose
    """
    print_banner()
    print("KARAOKE MODE")
    print("=" * 60)

    # Step 1: Get the audio file
    if source.startswith('http'):
        print("\nStep 1: Downloading from YouTube...")
        input_file = download_from_youtube(source)
    else:
        input_file = source
        if not Path(input_file).exists():
            print(f"Error: File not found: {input_file}")
            sys.exit(1)
        print(f"\nStep 1: Using local file: {input_file}")

    # Step 2: Remove vocals
    print("\nStep 2: Removing vocals...")
    instrumental = remove_vocals(input_file)

    # Step 3: Transpose if requested
    if semitones != 0:
        print(f"\nStep 3: Transposing by {semitones:+} semitones...")
        final_output = transpose_audio(instrumental, semitones, output_file)
    else:
        final_output = instrumental

    print("\n" + "=" * 60)
    print("KARAOKE TRACK READY!")
    print(f"Output: {final_output}")
    print("=" * 60)

    return final_output


def main():
    parser = argparse.ArgumentParser(
        description='Music Studio - Karaoke maker, stem separator, and transposer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  karaoke    Download and/or remove vocals for karaoke (most common use)
  vocals     Remove vocals to create instrumental
  isolate    Extract only the vocals (acapella)
  stems      Separate into all stems (drums, bass, vocals, other)
  transpose  Pitch shift audio
  download   Just download from YouTube

Examples:
  %(prog)s karaoke "https://youtube.com/watch?v=..."    Create karaoke track from YouTube
  %(prog)s karaoke song.mp3                             Remove vocals from local file
  %(prog)s karaoke "URL" -s 2                           Karaoke + transpose up 2 semitones
  %(prog)s vocals song.mp3                              Just remove vocals
  %(prog)s isolate song.mp3                             Extract just the vocals
  %(prog)s stems song.mp3                               Get all 4 stems separately
  %(prog)s transpose song.mp3 -s -3                     Pitch down 3 semitones
  %(prog)s download "URL"                               Just download from YouTube

Semitone reference:
  +1  = half step up       -1  = half step down
  +2  = whole step up      -2  = whole step down
  +12 = one octave up      -12 = one octave down
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Karaoke command
    karaoke_parser = subparsers.add_parser('karaoke', help='Create karaoke track (download + remove vocals)')
    karaoke_parser.add_argument('source', help='YouTube URL or local audio file')
    karaoke_parser.add_argument('-s', '--semitones', type=float, default=0,
                                help='Transpose by N semitones (optional)')
    karaoke_parser.add_argument('-o', '--output', help='Output file path')

    # Vocals (remove) command
    vocals_parser = subparsers.add_parser('vocals', help='Remove vocals from audio')
    vocals_parser.add_argument('input', help='Input audio file')
    vocals_parser.add_argument('-o', '--output', help='Output file path')

    # Isolate vocals command
    isolate_parser = subparsers.add_parser('isolate', help='Extract only vocals')
    isolate_parser.add_argument('input', help='Input audio file')
    isolate_parser.add_argument('-o', '--output', help='Output file path')

    # Stems command
    stems_parser = subparsers.add_parser('stems', help='Separate into all stems')
    stems_parser.add_argument('input', help='Input audio file')
    stems_parser.add_argument('-o', '--output-dir', help='Output directory')

    # Transpose command
    transpose_parser = subparsers.add_parser('transpose', help='Pitch shift audio')
    transpose_parser.add_argument('input', help='Input audio file')
    transpose_parser.add_argument('-s', '--semitones', type=float, required=True,
                                  help='Semitones to shift')
    transpose_parser.add_argument('-o', '--output', help='Output file path')
    transpose_parser.add_argument('--no-tempo-preserve', action='store_true',
                                  help='Allow tempo change with pitch')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download from YouTube')
    download_parser.add_argument('url', help='YouTube URL')
    download_parser.add_argument('-o', '--output-dir', help='Output directory')

    args = parser.parse_args()

    if not args.command:
        print_banner()
        parser.print_help()
        sys.exit(0)

    # Check dependencies based on command
    need_demucs = args.command in ['karaoke', 'vocals', 'isolate', 'stems']
    check_dependencies(need_demucs=need_demucs)

    # Execute command
    if args.command == 'karaoke':
        karaoke_mode(args.source, args.semitones, args.output)

    elif args.command == 'vocals':
        remove_vocals(args.input, args.output)

    elif args.command == 'isolate':
        extract_vocals_only(args.input, args.output)

    elif args.command == 'stems':
        separate_stems(args.input, args.output_dir)

    elif args.command == 'transpose':
        check_dependencies(need_demucs=False)
        transpose_audio(args.input, args.semitones, args.output,
                       speed_compensation=not args.no_tempo_preserve)

    elif args.command == 'download':
        download_from_youtube(args.url, args.output_dir)


if __name__ == '__main__':
    main()
