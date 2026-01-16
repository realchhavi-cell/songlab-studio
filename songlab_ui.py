#!/usr/bin/env python3
"""
SongLab - Your Personal Music Studio
Web UI for vocal removal, stem separation, and audio transposition
"""

import gradio as gr
import subprocess
import os
import shutil
from pathlib import Path
import tempfile

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


def download_from_youtube(url, progress=gr.Progress()):
    """Download audio from YouTube URL."""
    if not url or not url.strip():
        return None, "Please enter a YouTube URL"

    ytdlp = get_ytdlp()
    if not ytdlp:
        return None, "Error: yt-dlp not found. Please install it."

    progress(0.1, desc="Starting download...")

    # Create a safe filename
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
        return None, f"Download failed: {result.stderr}"

    downloaded_file = result.stdout.strip().split('\n')[-1]
    progress(1.0, desc="Download complete!")

    return downloaded_file, f"Downloaded: {Path(downloaded_file).name}"


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

    # Get vocals and create instrumental
    vocals_idx = stem_names.index('vocals')
    vocals = sources[vocals_idx]

    instrumental = None
    for i, stem_name in enumerate(stem_names):
        if stem_name != 'vocals':
            if instrumental is None:
                instrumental = sources[i]
            else:
                instrumental = instrumental + sources[i]

    # Save files
    temp_wav_inst = output_instrumental.with_suffix('.wav')
    temp_wav_voc = output_vocals.with_suffix('.wav')

    torchaudio.save(str(temp_wav_inst), instrumental, sr)
    torchaudio.save(str(temp_wav_voc), vocals, sr)

    # Convert to mp3
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
    stem_names = model.sources  # ['drums', 'bass', 'other', 'vocals']

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

    # Get sample rate
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
                 str(audio_file)]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    sample_rate = int(probe_result.stdout.strip()) if probe_result.stdout.strip() else 44100

    # Calculate pitch shift
    pitch_multiplier = 2 ** (semitones / 12)
    new_rate = int(sample_rate * pitch_multiplier)
    tempo_factor = 1 / pitch_multiplier

    # Build atempo filter chain
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


def download_and_separate(url, progress=gr.Progress()):
    """Download from YouTube and separate vocals in one step."""
    progress(0.1, desc="Downloading from YouTube...")
    audio_file, msg = download_from_youtube(url, progress)

    if audio_file is None:
        return None, None, None, msg

    progress(0.4, desc="Separating vocals...")
    instrumental, vocals, result_msg = remove_vocals(audio_file, progress)

    return audio_file, instrumental, vocals, f"Downloaded and separated: {Path(audio_file).name}"


# Build the Gradio UI
with gr.Blocks(title="SongLab - Music Studio") as app:

    gr.HTML("""
    <div class="main-title">
        <h1>SongLab</h1>
        <p>Your Personal Music Studio - Karaoke Maker, Stem Separator & Transposer</p>
    </div>
    """)

    with gr.Tabs():
        # Tab 1: Quick Karaoke (Download + Remove Vocals)
        with gr.TabItem("Karaoke Maker"):
            gr.Markdown("### Create karaoke tracks from YouTube or local files")

            with gr.Row():
                with gr.Column():
                    yt_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1
                    )
                    karaoke_btn = gr.Button("Create Karaoke Track", variant="primary", size="lg")

                    gr.Markdown("**Or upload a local file:**")
                    local_file = gr.Audio(label="Upload Audio File", type="filepath")
                    local_karaoke_btn = gr.Button("Remove Vocals from File", variant="secondary")

                with gr.Column():
                    karaoke_status = gr.Textbox(label="Status", interactive=False)
                    original_audio = gr.Audio(label="Original", type="filepath")
                    instrumental_audio = gr.Audio(label="Instrumental (Karaoke)", type="filepath")
                    vocals_audio = gr.Audio(label="Vocals", type="filepath")

            karaoke_btn.click(
                fn=download_and_separate,
                inputs=[yt_url],
                outputs=[original_audio, instrumental_audio, vocals_audio, karaoke_status]
            )

            local_karaoke_btn.click(
                fn=remove_vocals,
                inputs=[local_file],
                outputs=[instrumental_audio, vocals_audio, karaoke_status]
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
                fn=download_from_youtube,
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
        <p>SongLab uses Demucs AI for high-quality audio separation</p>
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
        server_name="127.0.0.1",
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="pink")
    )
