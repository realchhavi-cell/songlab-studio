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

# Project directories
PROJECT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Demucs model
DEMUCS_MODEL = "htdemucs"

# Custom CSS for Apple Music / BandLab DAW style
CUSTOM_CSS = """
/* Main container */
.gradio-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%) !important;
    min-height: 100vh;
}

/* Header styling */
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}

.main-header h1 {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    margin-bottom: 0.5rem !important;
}

.main-header p {
    font-size: 1.2rem !important;
    opacity: 0.9;
    color: #a0a0a0 !important;
    -webkit-text-fill-color: #a0a0a0 !important;
}

/* Tab styling */
.tabs {
    background: transparent !important;
    border: none !important;
}

.tabitem {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    margin-top: 0.5rem !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.tab-nav {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 15px !important;
    padding: 0.5rem !important;
    gap: 0.5rem !important;
    border: none !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: #888 !important;
    font-weight: 600 !important;
    padding: 0.8rem 1.5rem !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.tab-nav button:hover:not(.selected) {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #fff !important;
}

/* Input fields */
.gr-input, .gr-text-input, input[type="text"], textarea {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: #fff !important;
    padding: 12px 16px !important;
    transition: all 0.3s ease !important;
}

.gr-input:focus, .gr-text-input:focus, input[type="text"]:focus, textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
    outline: none !important;
}

/* Buttons */
.gr-button, button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.gr-button:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5) !important;
}

.gr-button.secondary, button.secondary {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: none !important;
}

.gr-button.secondary:hover, button.secondary:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Sliders */
input[type="range"] {
    accent-color: #667eea !important;
}

.gr-slider, .slider {
    background: transparent !important;
}

.gr-slider input[type="range"]::-webkit-slider-track {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
    border-radius: 10px !important;
    height: 8px !important;
}

/* Audio players */
.gr-audio, audio {
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Labels */
label, .gr-label, .label-wrap {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Section titles */
.section-title {
    color: #fff !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

/* Cards */
.feature-card {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s ease !important;
}

.feature-card:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3) !important;
}

/* Status box */
.status-box {
    background: rgba(102, 126, 234, 0.1) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    color: #a0c4ff !important;
}

/* Progress bar */
.progress-bar {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.progress-bar .progress {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: #666;
    font-size: 0.9rem;
}

/* Markdown text */
.gr-markdown, .markdown-text {
    color: #c0c0c0 !important;
}

.gr-markdown h3, .markdown-text h3 {
    color: #fff !important;
    font-weight: 600 !important;
}

/* File upload area */
.gr-file, .gr-audio .upload-area {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 2px dashed rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}

.gr-file:hover, .gr-audio .upload-area:hover {
    border-color: #667eea !important;
    background: rgba(102, 126, 234, 0.05) !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2.5rem !important;
    }

    .tab-nav button {
        padding: 0.6rem 1rem !important;
        font-size: 0.9rem !important;
    }
}

/* Glow effects */
.glow-purple {
    box-shadow: 0 0 30px rgba(102, 126, 234, 0.3) !important;
}

.glow-pink {
    box-shadow: 0 0 30px rgba(240, 147, 251, 0.3) !important;
}

/* Animation for buttons */
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

.pulse-animation {
    animation: pulse 2s infinite;
}
"""


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
    st_str = f"{int(semitones)}" if semitones == int(semitones) else f"{semitones:.1f}"
    output_file = OUTPUT_DIR / f"{input_path.stem}_key_{sign}{st_str}.mp3"

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

    return str(output_file), f"Transposed by {semitones:+g} semitones"


def karaoke_full_pipeline(url, local_file, semitones, progress=gr.Progress()):
    """Full karaoke pipeline: Download (if URL) -> Remove vocals -> Transpose (if needed)"""

    # Step 1: Get the audio file
    if url and url.strip():
        progress(0.1, desc="Downloading from YouTube...")
        audio_file, msg = download_from_youtube(url, progress)
        if audio_file is None:
            return None, None, None, None, msg
    elif local_file:
        audio_file = local_file
    else:
        return None, None, None, None, "Please provide a YouTube URL or upload a file"

    # Step 2: Remove vocals
    progress(0.3, desc="Removing vocals with AI...")
    instrumental, vocals, msg = remove_vocals(audio_file, progress)

    if instrumental is None:
        return audio_file, None, None, None, msg

    # Step 3: Transpose if needed
    final_instrumental = instrumental
    if semitones != 0:
        progress(0.8, desc=f"Transposing by {semitones:+g} semitones...")
        final_instrumental, transpose_msg = transpose_audio(instrumental, semitones, progress)

    progress(1.0, desc="Karaoke track ready!")

    return audio_file, final_instrumental, vocals, instrumental, "Karaoke track created successfully!"


# Build the Gradio UI
with gr.Blocks(title="SongLab - Music Studio", css=CUSTOM_CSS) as app:

    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>SongLab</h1>
        <p>Your AI-Powered Music Studio</p>
    </div>
    """)

    with gr.Tabs() as tabs:

        # Tab 1: Karaoke Studio (Main)
        with gr.TabItem("Karaoke Studio", id=0):
            gr.HTML('<div class="section-title">Create Karaoke Tracks Instantly</div>')

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p style="color: #888; margin-bottom: 1rem;">Paste a YouTube link or drop an audio file</p>')

                    yt_url_main = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1,
                        elem_classes=["glow-purple"]
                    )

                    gr.HTML('<p style="color: #666; text-align: center; margin: 0.5rem 0;">— or —</p>')

                    local_file_main = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        elem_classes=["feature-card"]
                    )

                    gr.HTML('<div style="margin-top: 1.5rem;"><p style="color: #888; margin-bottom: 0.5rem;">Adjust Key (Transpose)</p></div>')

                    semitones_main = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=1,
                        label="Semitones",
                        info="Shift pitch up or down to match your vocal range"
                    )

                    gr.HTML("""
                    <div style="display: flex; justify-content: space-between; color: #666; font-size: 0.8rem; margin-top: -0.5rem;">
                        <span>-12 (Lower)</span>
                        <span>0 (Original)</span>
                        <span>+12 (Higher)</span>
                    </div>
                    """)

                    create_karaoke_btn = gr.Button(
                        "Create Karaoke Track",
                        variant="primary",
                        size="lg",
                        elem_classes=["pulse-animation"]
                    )

                    status_main = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes=["status-box"]
                    )

                with gr.Column(scale=1):
                    gr.HTML('<p style="color: #888; margin-bottom: 1rem;">Your processed tracks</p>')

                    with gr.Group(elem_classes=["feature-card"]):
                        original_out = gr.Audio(label="Original Track", type="filepath")

                    with gr.Group(elem_classes=["feature-card", "glow-purple"]):
                        instrumental_out = gr.Audio(label="Instrumental (Karaoke)", type="filepath")

                    with gr.Group(elem_classes=["feature-card"]):
                        vocals_out = gr.Audio(label="Isolated Vocals", type="filepath")

                    # Hidden output for non-transposed instrumental
                    instrumental_original = gr.Audio(visible=False, type="filepath")

            create_karaoke_btn.click(
                fn=karaoke_full_pipeline,
                inputs=[yt_url_main, local_file_main, semitones_main],
                outputs=[original_out, instrumental_out, vocals_out, instrumental_original, status_main]
            )

        # Tab 2: Stem Separator
        with gr.TabItem("Stem Separator", id=1):
            gr.HTML('<div class="section-title">Split Into Individual Tracks</div>')
            gr.HTML('<p style="color: #888; margin-bottom: 1rem;">Separate any song into drums, bass, vocals, and other instruments</p>')

            with gr.Row():
                with gr.Column(scale=1):
                    stem_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        elem_classes=["feature-card"]
                    )
                    stem_btn = gr.Button("Separate Stems", variant="primary", size="lg")
                    stem_status = gr.Textbox(label="Status", interactive=False, elem_classes=["status-box"])

                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Column():
                            drums_audio = gr.Audio(label="Drums", type="filepath", elem_classes=["feature-card"])
                        with gr.Column():
                            bass_audio = gr.Audio(label="Bass", type="filepath", elem_classes=["feature-card"])
                    with gr.Row():
                        with gr.Column():
                            vocals_stem = gr.Audio(label="Vocals", type="filepath", elem_classes=["feature-card"])
                        with gr.Column():
                            other_audio = gr.Audio(label="Other", type="filepath", elem_classes=["feature-card"])

            stem_btn.click(
                fn=separate_stems,
                inputs=[stem_input],
                outputs=[drums_audio, bass_audio, vocals_stem, other_audio, stem_status]
            )

        # Tab 3: Transpose
        with gr.TabItem("Transpose", id=2):
            gr.HTML('<div class="section-title">Change Key / Pitch</div>')
            gr.HTML('<p style="color: #888; margin-bottom: 1rem;">Shift any audio up or down to match your vocal range</p>')

            with gr.Row():
                with gr.Column(scale=1):
                    transpose_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        elem_classes=["feature-card"]
                    )

                    semitones_slider = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=1,
                        label="Semitones"
                    )

                    gr.HTML("""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; margin: 1rem 0;">
                        <p style="color: #888; font-size: 0.9rem; margin: 0;">
                            <strong style="color: #fff;">Quick Reference:</strong><br>
                            ±1 = Half step &nbsp;|&nbsp; ±2 = Whole step &nbsp;|&nbsp; ±12 = One octave
                        </p>
                    </div>
                    """)

                    transpose_btn = gr.Button("Transpose Audio", variant="primary", size="lg")

                with gr.Column(scale=1):
                    transpose_status = gr.Textbox(label="Status", interactive=False, elem_classes=["status-box"])
                    transposed_audio = gr.Audio(
                        label="Transposed Audio",
                        type="filepath",
                        elem_classes=["feature-card", "glow-pink"]
                    )

            transpose_btn.click(
                fn=transpose_audio,
                inputs=[transpose_input, semitones_slider],
                outputs=[transposed_audio, transpose_status]
            )

        # Tab 4: YouTube Download
        with gr.TabItem("YouTube Download", id=3):
            gr.HTML('<div class="section-title">Download from YouTube</div>')
            gr.HTML('<p style="color: #888; margin-bottom: 1rem;">Extract high-quality audio from any YouTube video</p>')

            with gr.Row():
                with gr.Column(scale=1):
                    dl_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1
                    )
                    dl_btn = gr.Button("Download Audio", variant="primary", size="lg")

                with gr.Column(scale=1):
                    dl_status = gr.Textbox(label="Status", interactive=False, elem_classes=["status-box"])
                    dl_audio = gr.Audio(
                        label="Downloaded Audio",
                        type="filepath",
                        elem_classes=["feature-card", "glow-purple"]
                    )

            dl_btn.click(
                fn=download_from_youtube,
                inputs=[dl_url],
                outputs=[dl_audio, dl_status]
            )

    # Footer
    gr.HTML("""
    <div class="footer">
        <p style="color: #666;">Powered by Demucs AI | Made with Gradio</p>
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
