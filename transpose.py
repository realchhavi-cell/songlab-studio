#!/usr/bin/env python3
"""
Transpose App - Pitch shift audio/video files and save them
Similar to the Chrome Transpose extension, but with the ability to save!

Usage:
    python transpose.py input.mp3 -s 3           # Shift up 3 semitones
    python transpose.py input.mp4 -s -2          # Shift down 2 semitones
    python transpose.py -y "youtube_url" -s 4    # Download from YouTube & transpose
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path

# Project directory - all outputs go here
PROJECT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_DIR / "output"


def check_dependencies():
    """Check if required tools are installed."""
    missing = []

    if not shutil.which('ffmpeg'):
        missing.append('ffmpeg')

    if not shutil.which('yt-dlp') and not shutil.which('youtube-dl'):
        missing.append('yt-dlp')

    if missing:
        print("Missing dependencies:", ', '.join(missing))
        print("\nInstall them with:")
        print("  brew install ffmpeg yt-dlp")
        print("\nOr on other systems:")
        print("  # FFmpeg: https://ffmpeg.org/download.html")
        print("  # yt-dlp: pip install yt-dlp")
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

    # Always save to project output directory
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, '%(title)s.%(ext)s')

    print(f"Downloading from YouTube: {url}")

    cmd = [
        ytdlp,
        '-x',  # Extract audio
        '--audio-format', 'mp3',
        '--audio-quality', '0',  # Best quality
        '-o', output_template,
        '--print', 'after_move:filepath',  # Print the final filename
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        sys.exit(1)

    # Get the downloaded filename from output
    downloaded_file = result.stdout.strip().split('\n')[-1]
    print(f"Downloaded: {downloaded_file}")
    return downloaded_file


def transpose_audio(input_file, semitones, output_file=None, speed_compensation=True):
    """
    Transpose audio by the specified number of semitones.

    Args:
        input_file: Path to input audio/video file
        semitones: Number of semitones to shift (positive = higher, negative = lower)
        output_file: Optional output path (default: adds _transposed_Xst suffix)
        speed_compensation: If True, maintains original tempo (like the Transpose extension)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    # Generate output filename - save to project output directory
    if not output_file:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sign = "+" if semitones >= 0 else ""
        # Format semitones nicely (remove .0 for whole numbers)
        st_str = f"{semitones:.1f}".rstrip('0').rstrip('.')
        suffix = f"_transposed_{sign}{st_str}st"
        output_file = OUTPUT_DIR / f"{input_path.stem}{suffix}{input_path.suffix}"

    output_path = Path(output_file)

    # Calculate pitch multiplier
    # Each semitone is a factor of 2^(1/12)
    pitch_multiplier = 2 ** (semitones / 12)

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Shift:  {semitones:+.1f} semitones (pitch multiplier: {pitch_multiplier:.4f})")

    # Check if input is video
    is_video = input_path.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']

    # Get input sample rate
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
                 str(input_path)]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    sample_rate = int(probe_result.stdout.strip()) if probe_result.stdout.strip() else 44100

    if speed_compensation:
        # Pitch shift while maintaining tempo using asetrate + atempo
        # asetrate changes pitch (and tempo), then atempo corrects the tempo back
        new_rate = int(sample_rate * pitch_multiplier)

        # Compensate for tempo change with atempo
        # If we pitch up (multiplier > 1), audio gets faster, so slow it down
        tempo_factor = 1 / pitch_multiplier

        # atempo only accepts values between 0.5 and 2.0
        # Chain multiple atempo filters if needed for extreme shifts
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
        # Simple pitch shift (will also change tempo)
        new_rate = int(sample_rate * pitch_multiplier)
        filter_complex = f"asetrate={new_rate},aresample={sample_rate}"

    if is_video:
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-af', filter_complex,
            '-c:v', 'copy',  # Copy video without re-encoding
            str(output_path)
        ]
    else:
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-af', filter_complex,
            str(output_path)
        ]

    print("\nProcessing...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(f"\nDone! Transposed file saved to:\n  {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Transpose (pitch shift) audio/video files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.mp3 -s 3           Shift up 3 semitones
  %(prog)s song.mp3 -s -2          Shift down 2 semitones
  %(prog)s video.mp4 -s 5          Transpose video audio
  %(prog)s -y "URL" -s 4           Download from YouTube and transpose
  %(prog)s song.mp3 -s 3 -o out.mp3  Specify output filename

Semitone reference:
  +1  = half step up      -1  = half step down
  +2  = whole step up     -2  = whole step down
  +12 = one octave up     -12 = one octave down
        """
    )

    parser.add_argument('input', nargs='?', help='Input audio/video file')
    parser.add_argument('-s', '--semitones', type=float, required=True,
                        help='Number of semitones to shift (positive=higher, negative=lower)')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-y', '--youtube', help='YouTube URL to download and transpose')
    parser.add_argument('--no-tempo-preserve', action='store_true',
                        help='Allow tempo to change with pitch (faster processing)')

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Get input file
    if args.youtube:
        input_file = download_from_youtube(args.youtube)
    elif args.input:
        input_file = args.input
    else:
        parser.error("Either provide an input file or use -y for YouTube URL")

    # Transpose
    transpose_audio(
        input_file,
        args.semitones,
        args.output,
        speed_compensation=not args.no_tempo_preserve
    )


if __name__ == '__main__':
    main()
