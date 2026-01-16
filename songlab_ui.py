#!/usr/bin/env python3
"""
SongLab - Your Personal Music Studio
Web UI for vocal removal, stem separation, and audio transposition
"""

import gradio as gr
import subprocess
import os
import shutil
import re
from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Demucs model
DEMUCS_MODEL = "htdemucs"


def get_ytdlp():
    """Get the YouTube downloader command."""
    if shutil.which('yt-dlp'):
        return 'yt-dlp'
    elif shutil.which('youtube-dl'):
        return 'youtube-dl'
    return None


def fetch_lyrics(song_title, artist=""):
    """Fetch synced lyrics for a song."""
    try:
        import syncedlyrics

        search_query = f"{song_title} {artist}".strip()

        # Try to get synced (LRC) lyrics first
        lrc = syncedlyrics.search(search_query, synced_only=True)

        if lrc:
            return lrc, "synced"

        # Fall back to plain lyrics
        plain = syncedlyrics.search(search_query, synced_only=False)
        if plain:
            return plain, "plain"

        return None, None
    except Exception as e:
        print(f"Lyrics fetch error: {e}")
        return None, None


def parse_lrc_to_html(lrc_text):
    """Convert LRC format to HTML with timestamps for display."""
    if not lrc_text:
        return ""

    lines = lrc_text.strip().split('\n')
    html_lines = []

    # LRC timestamp pattern: [mm:ss.xx] or [mm:ss]
    timestamp_pattern = re.compile(r'\[(\d{2}):(\d{2})(?:\.(\d{2,3}))?\]')

    for line in lines:
        # Skip metadata lines like [ar:Artist]
        if line.startswith('[') and ':' in line and not timestamp_pattern.match(line):
            continue

        # Extract timestamp and text
        match = timestamp_pattern.match(line)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            total_seconds = minutes * 60 + seconds
            text = timestamp_pattern.sub('', line).strip()

            if text:  # Only add non-empty lines
                html_lines.append(f'<div class="lyric-line" data-time="{total_seconds}">'
                                  f'<span class="timestamp">[{minutes:02d}:{seconds:02d}]</span> '
                                  f'<span class="text">{text}</span></div>')
        else:
            # Plain text line
            text = line.strip()
            if text and not text.startswith('['):
                html_lines.append(f'<div class="lyric-line"><span class="text">{text}</span></div>')

    return '\n'.join(html_lines)


def format_lyrics_display(lrc_text, lyrics_type):
    """Format lyrics for display in Gradio."""
    if not lrc_text:
        return "No lyrics found. Try searching manually with artist name."

    if lyrics_type == "synced":
        # Parse LRC and create formatted display
        lines = lrc_text.strip().split('\n')
        formatted_lines = []

        timestamp_pattern = re.compile(r'\[(\d{2}):(\d{2})(?:\.(\d{2,3}))?\]')

        for line in lines:
            # Skip metadata
            if line.startswith('[') and ':' in line and not timestamp_pattern.match(line):
                continue

            match = timestamp_pattern.match(line)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                text = timestamp_pattern.sub('', line).strip()
                if text:
                    formatted_lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")
            else:
                text = line.strip()
                if text and not text.startswith('['):
                    formatted_lines.append(text)

        return '\n'.join(formatted_lines)
    else:
        # Plain lyrics - just clean up
        return lrc_text


def extract_song_info_from_filename(filename):
    """Try to extract song title and artist from filename."""
    if not filename:
        return "", ""

    name = Path(filename).stem

    # Common patterns: "Artist - Song", "Song - Artist", "Song (Artist)"
    # Clean up common suffixes
    name = re.sub(r'_instrumental|_vocals|_transposed.*|\(Official.*\)|\(Lyric.*\)|【.*】|\[.*\]', '', name, flags=re.IGNORECASE)
    name = name.strip()

    # Try to split by " - "
    if ' - ' in name:
        parts = name.split(' - ', 1)
        return parts[1].strip(), parts[0].strip()  # Assume "Artist - Song"

    return name, ""


def search_lyrics(song_title, artist_name):
    """Search for lyrics with given title and artist."""
    if not song_title:
        return "Please enter a song title"

    lyrics, lyrics_type = fetch_lyrics(song_title, artist_name)

    if lyrics:
        formatted = format_lyrics_display(lyrics, lyrics_type)
        status = "Synced lyrics found!" if lyrics_type == "synced" else "Plain lyrics found (no timestamps)"
        return formatted
    else:
        return "No lyrics found. Try different spelling or add artist name."


def download_from_youtube(url, progress=gr.Progress()):
    """Download audio from YouTube URL."""
    if not url or not url.strip():
        return None, "Please enter a YouTube URL", ""

    ytdlp = get_ytdlp()
    if not ytdlp:
        return None, "Error: yt-dlp not found. Please install it.", ""

    progress(0.1, desc="Starting download...")

    output_template = str(OUTPUT_DIR / '%(title)s.%(ext)s')

    cmd = [
        ytdlp,
        '-x',
        '--audio-format', 'mp3',
        '--audio-quality', '0',
        '-o', output_template,
        '--print', 'after_move:filepath',
        url.strip()
    ]

    progress(0.3, desc="Downloading from YouTube...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return None, f"Download failed: {result.stderr}", ""

    downloaded_file = result.stdout.strip().split('\n')[-1]
    song_name = Path(downloaded_file).stem

    progress(1.0, desc="Download complete!")

    return downloaded_file, f"Downloaded: {song_name}", song_name


def remove_vocals(audio_file, progress=gr.Progress()):
    """Remove vocals from audio to create instrumental."""
    if audio_file is None:
        return None, None, "Please provide an audio file"

    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    input_path = Path(audio_file)
    output_instrumental = OUTPUT_DIR / f"{input_path.stem}_instrumental.mp3"
    output_vocals = OUTPUT_DIR / f"{input_path.stem}_vocals.mp3"

    progress(0.1, desc="Loading AI model...")

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = get_model(DEMUCS_MODEL)
    model.to(device)
    model.eval()

    progress(0.2, desc="Loading audio...")
    wav, sr = torchaudio.load(audio_file)

    if sr != model.samplerate:
        resampler = torchaudio.transforms.Resample(sr, model.samplerate)
        wav = resampler(wav)
        sr = model.samplerate

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    wav = wav.unsqueeze(0).to(device)

    progress(0.3, desc="Separating vocals (this takes a moment)...")
    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=False)

    progress(0.8, desc="Creating output files...")
    sources = sources.squeeze(0).cpu()
    stem_names = model.sources

    vocals_idx = stem_names.index('vocals')
    vocals = sources[vocals_idx]

    instrumental = None
    for i, stem_name in enumerate(stem_names):
        if stem_name != 'vocals':
            if instrumental is None:
                instrumental = sources[i]
            else:
                instrumental = instrumental + sources[i]

    temp_wav_inst = output_instrumental.with_suffix('.wav')
    temp_wav_voc = output_vocals.with_suffix('.wav')

    torchaudio.save(str(temp_wav_inst), instrumental, sr)
    torchaudio.save(str(temp_wav_voc), vocals, sr)

    subprocess.run(['ffmpeg', '-y', '-i', str(temp_wav_inst), '-acodec', 'libmp3lame', '-q:a', '2', str(output_instrumental)], capture_output=True)
    subprocess.run(['ffmpeg', '-y', '-i', str(temp_wav_voc), '-acodec', 'libmp3lame', '-q:a', '2', str(output_vocals)], capture_output=True)

    temp_wav_inst.unlink()
    temp_wav_voc.unlink()

    progress(1.0, desc="Done!")

    return str(output_instrumental), str(output_vocals), "Vocal separation complete!"


def separate_stems(audio_file, progress=gr.Progress()):
    """Separate audio into 4 stems: drums, bass, vocals, other."""
    if audio_file is None:
        return None, None, None, None, "Please provide an audio file"

    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    input_path = Path(audio_file)
    stem_dir = OUTPUT_DIR / f"{input_path.stem}_stems"
    os.makedirs(stem_dir, exist_ok=True)

    progress(0.1, desc="Loading AI model...")

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = get_model(DEMUCS_MODEL)
    model.to(device)
    model.eval()

    progress(0.2, desc="Loading audio...")
    wav, sr = torchaudio.load(audio_file)

    if sr != model.samplerate:
        resampler = torchaudio.transforms.Resample(sr, model.samplerate)
        wav = resampler(wav)
        sr = model.samplerate

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    wav = wav.unsqueeze(0).to(device)

    progress(0.3, desc="Separating stems (this takes a moment)...")
    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=False)

    progress(0.8, desc="Saving stem files...")
    sources = sources.squeeze(0).cpu()
    stem_names = model.sources

    stem_files = {}
    for i, stem_name in enumerate(stem_names):
        stem_path = stem_dir / f"{stem_name}.mp3"
        temp_wav = stem_dir / f"{stem_name}_temp.wav"

        torchaudio.save(str(temp_wav), sources[i], sr)
        subprocess.run(['ffmpeg', '-y', '-i', str(temp_wav), '-acodec', 'libmp3lame', '-q:a', '2', str(stem_path)], capture_output=True)
        temp_wav.unlink()

        stem_files[stem_name] = str(stem_path)

    progress(1.0, desc="Done!")

    return (
        stem_files.get('drums'),
        stem_files.get('bass'),
        stem_files.get('vocals'),
        stem_files.get('other'),
        "Stem separation complete!"
    )


def transpose_audio(audio_file, semitones, progress=gr.Progress()):
    """Transpose audio by specified semitones."""
    if audio_file is None:
        return None, "Please provide an audio file"

    if semitones == 0:
        return audio_file, "No transposition needed (0 semitones)"

    input_path = Path(audio_file)
    sign = "+" if semitones >= 0 else ""
    st_str = f"{semitones:.1f}".rstrip('0').rstrip('.')
    output_file = OUTPUT_DIR / f"{input_path.stem}_transposed_{sign}{st_str}st.mp3"

    progress(0.2, desc="Analyzing audio...")

    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
                 str(audio_file)]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    sample_rate = int(probe_result.stdout.strip()) if probe_result.stdout.strip() else 44100

    pitch_multiplier = 2 ** (semitones / 12)
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

    progress(0.5, desc="Transposing audio...")

    cmd = ['ffmpeg', '-y', '-i', str(audio_file), '-af', filter_complex, str(output_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return None, f"Error: {result.stderr}"

    progress(1.0, desc="Done!")

    return str(output_file), f"Transposed by {semitones:+} semitones"


def download_and_separate_with_lyrics(url, progress=gr.Progress()):
    """Download from YouTube, separate vocals, and fetch lyrics."""
    progress(0.1, desc="Downloading from YouTube...")
    audio_file, msg, song_name = download_from_youtube(url, progress)

    if audio_file is None:
        return None, None, None, msg, "", ""

    progress(0.3, desc="Fetching lyrics...")
    song_title, artist = extract_song_info_from_filename(audio_file)
    lyrics, lyrics_type = fetch_lyrics(song_title, artist)
    formatted_lyrics = format_lyrics_display(lyrics, lyrics_type) if lyrics else "No lyrics found. Search manually below."

    progress(0.4, desc="Separating vocals...")
    instrumental, vocals, result_msg = remove_vocals(audio_file, progress)

    return audio_file, instrumental, vocals, f"Downloaded and separated: {song_name}", formatted_lyrics, song_name


def process_local_file_with_lyrics(audio_file, progress=gr.Progress()):
    """Remove vocals from local file and try to fetch lyrics."""
    if audio_file is None:
        return None, None, "Please provide an audio file", "", ""

    song_title, artist = extract_song_info_from_filename(audio_file)

    progress(0.1, desc="Fetching lyrics...")
    lyrics, lyrics_type = fetch_lyrics(song_title, artist)
    formatted_lyrics = format_lyrics_display(lyrics, lyrics_type) if lyrics else "No lyrics found. Search manually below."

    progress(0.2, desc="Removing vocals...")
    instrumental, vocals, status = remove_vocals(audio_file, progress)

    return instrumental, vocals, status, formatted_lyrics, song_title


# Build the Gradio UI
with gr.Blocks(title="SongLab - Music Studio", theme=gr.themes.Soft(primary_hue="purple", secondary_hue="pink")) as app:

    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">SongLab</h1>
        <p style="color: #666;">Your Personal Music Studio - Karaoke Maker with Lyrics</p>
    </div>
    """)

    with gr.Tabs():
        # Tab 1: Karaoke Maker with Lyrics
        with gr.TabItem("Karaoke Maker"):
            gr.Markdown("### Create karaoke tracks with synchronized lyrics")

            with gr.Row():
                with gr.Column(scale=1):
                    yt_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1
                    )
                    karaoke_btn = gr.Button("Create Karaoke Track", variant="primary", size="lg")

                    gr.Markdown("**Or upload a local file:**")
                    local_file = gr.Audio(label="Upload Audio File", type="filepath")
                    local_karaoke_btn = gr.Button("Remove Vocals from File", variant="secondary")

                    karaoke_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=1):
                    original_audio = gr.Audio(label="Original", type="filepath")
                    instrumental_audio = gr.Audio(label="Instrumental (Karaoke)", type="filepath")
                    vocals_audio = gr.Audio(label="Vocals", type="filepath")

            gr.Markdown("---")
            gr.Markdown("### Lyrics")

            with gr.Row():
                with gr.Column(scale=1):
                    song_title_input = gr.Textbox(label="Song Title", placeholder="Enter song title...")
                    artist_input = gr.Textbox(label="Artist (optional)", placeholder="Enter artist name...")
                    search_lyrics_btn = gr.Button("Search Lyrics", variant="secondary")

                with gr.Column(scale=2):
                    lyrics_display = gr.Textbox(
                        label="Lyrics",
                        lines=15,
                        max_lines=25,
                        interactive=False,
                        placeholder="Lyrics will appear here after processing...\n\nTimestamps show when each line should be sung."
                    )

            # Connect buttons
            karaoke_btn.click(
                fn=download_and_separate_with_lyrics,
                inputs=[yt_url],
                outputs=[original_audio, instrumental_audio, vocals_audio, karaoke_status, lyrics_display, song_title_input]
            )

            local_karaoke_btn.click(
                fn=process_local_file_with_lyrics,
                inputs=[local_file],
                outputs=[instrumental_audio, vocals_audio, karaoke_status, lyrics_display, song_title_input]
            )

            search_lyrics_btn.click(
                fn=search_lyrics,
                inputs=[song_title_input, artist_input],
                outputs=[lyrics_display]
            )

        # Tab 2: YouTube Download
        with gr.TabItem("YouTube Download"):
            gr.Markdown("### Download audio from YouTube")

            with gr.Row():
                with gr.Column():
                    dl_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1
                    )
                    dl_btn = gr.Button("Download", variant="primary")

                with gr.Column():
                    dl_status = gr.Textbox(label="Status", interactive=False)
                    dl_audio = gr.Audio(label="Downloaded Audio", type="filepath")

            dl_btn.click(
                fn=lambda url: download_from_youtube(url)[:2],
                inputs=[dl_url],
                outputs=[dl_audio, dl_status]
            )

        # Tab 3: Stem Separation
        with gr.TabItem("Stem Separator"):
            gr.Markdown("### Separate audio into drums, bass, vocals, and other instruments")

            with gr.Row():
                with gr.Column():
                    stem_input = gr.Audio(label="Upload Audio File", type="filepath")
                    stem_btn = gr.Button("Separate Stems", variant="primary")
                    stem_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column():
                    drums_audio = gr.Audio(label="Drums", type="filepath")
                    bass_audio = gr.Audio(label="Bass", type="filepath")
                    vocals_stem = gr.Audio(label="Vocals", type="filepath")
                    other_audio = gr.Audio(label="Other (guitars, synths, etc.)", type="filepath")

            stem_btn.click(
                fn=separate_stems,
                inputs=[stem_input],
                outputs=[drums_audio, bass_audio, vocals_stem, other_audio, stem_status]
            )

        # Tab 4: Transpose
        with gr.TabItem("Transpose"):
            gr.Markdown("### Change the key/pitch of your audio")

            with gr.Row():
                with gr.Column():
                    transpose_input = gr.Audio(label="Upload Audio File", type="filepath")
                    semitones_slider = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=1,
                        label="Semitones (negative = lower, positive = higher)"
                    )
                    gr.Markdown("""
                    **Quick reference:**
                    - ±1 = half step
                    - ±2 = whole step
                    - ±12 = one octave
                    """)
                    transpose_btn = gr.Button("Transpose", variant="primary")

                with gr.Column():
                    transpose_status = gr.Textbox(label="Status", interactive=False)
                    transposed_audio = gr.Audio(label="Transposed Audio", type="filepath")

            transpose_btn.click(
                fn=transpose_audio,
                inputs=[transpose_input, semitones_slider],
                outputs=[transposed_audio, transpose_status]
            )

    gr.HTML("""
    <div style="text-align: center; margin-top: 2rem; color: #666;">
        <p>SongLab uses Demucs AI for audio separation | Lyrics powered by syncedlyrics</p>
    </div>
    """)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SongLab - Your Personal Music Studio")
    print("="*60)
    print("\nStarting web interface...")
    print("Open your browser to the URL shown below\n")

    app.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1"
    )
