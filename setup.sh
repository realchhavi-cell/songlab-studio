#!/bin/bash
# Setup script for Music Studio App
# Installs required dependencies: ffmpeg, yt-dlp, and demucs (for vocal removal)

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  MUSIC STUDIO SETUP                          ║"
echo "║       Karaoke Maker | Stem Separator | Transposer            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
else
    OS="other"
fi

# Check for Homebrew on macOS
install_system_deps_brew() {
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    echo "Installing system dependencies with Homebrew..."
    brew install ffmpeg yt-dlp
}

# Check for apt on Linux
install_system_deps_apt() {
    echo "Installing system dependencies with apt..."
    sudo apt update
    sudo apt install -y ffmpeg python3-pip
    pip3 install yt-dlp
}

# Install Python dependencies (demucs for vocal removal)
install_python_deps() {
    echo ""
    echo "Installing Python dependencies for vocal removal..."
    echo "This includes PyTorch and Demucs (may take a few minutes)..."
    echo ""

    # Check if pip is available
    if command -v pip3 &> /dev/null; then
        PIP="pip3"
    elif command -v pip &> /dev/null; then
        PIP="pip"
    else
        echo "Error: pip not found. Please install Python and pip first."
        exit 1
    fi

    # Install demucs (this will also install torch and torchaudio)
    $PIP install demucs

    echo ""
    echo "Python dependencies installed!"
}

# Check if system dependencies are installed
check_system_deps() {
    local missing=0

    echo "System dependencies:"

    if command -v ffmpeg &> /dev/null; then
        echo "  [OK] ffmpeg"
    else
        echo "  [X]  ffmpeg - NOT FOUND"
        missing=1
    fi

    if command -v yt-dlp &> /dev/null || command -v youtube-dl &> /dev/null; then
        echo "  [OK] yt-dlp"
    else
        echo "  [X]  yt-dlp - NOT FOUND"
        missing=1
    fi

    return $missing
}

# Check if Python dependencies are installed
check_python_deps() {
    local missing=0

    echo ""
    echo "Python dependencies:"

    if python3 -c "import demucs" 2>/dev/null; then
        echo "  [OK] demucs (vocal removal)"
    else
        echo "  [X]  demucs - NOT FOUND"
        missing=1
    fi

    if python3 -c "import torch" 2>/dev/null; then
        echo "  [OK] torch (PyTorch)"
    else
        echo "  [X]  torch - NOT FOUND"
        missing=1
    fi

    if python3 -c "import torchaudio" 2>/dev/null; then
        echo "  [OK] torchaudio"
    else
        echo "  [X]  torchaudio - NOT FOUND"
        missing=1
    fi

    return $missing
}

# Check for GPU support
check_gpu() {
    echo ""
    echo "Hardware acceleration:"

    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "  [OK] CUDA GPU detected - processing will be faster!"
    elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        echo "  [OK] Apple Silicon MPS detected - processing will be faster!"
    else
        echo "  [--] CPU only - processing will work but may be slower"
    fi
}

echo "Checking dependencies..."
echo ""

SYSTEM_OK=0
PYTHON_OK=0

if check_system_deps; then
    SYSTEM_OK=1
fi

if check_python_deps; then
    PYTHON_OK=1
fi

if [[ $SYSTEM_OK -eq 1 && $PYTHON_OK -eq 1 ]]; then
    echo ""
    echo "All dependencies are already installed!"
    check_gpu
else
    echo ""

    # Install system dependencies if missing
    if [[ $SYSTEM_OK -eq 0 ]]; then
        echo "Some system dependencies are missing."
        echo ""

        if [[ "$OS" == "macos" ]]; then
            read -p "Install system dependencies with Homebrew? (y/n) " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                install_system_deps_brew
            fi
        elif [[ "$OS" == "linux" ]]; then
            read -p "Install system dependencies with apt? (y/n) " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                install_system_deps_apt
            fi
        else
            echo "Please install manually:"
            echo "  ffmpeg: https://ffmpeg.org/download.html"
            echo "  yt-dlp: pip install yt-dlp"
        fi
    fi

    # Install Python dependencies if missing
    if [[ $PYTHON_OK -eq 0 ]]; then
        echo ""
        read -p "Install Python dependencies (demucs for vocal removal)? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_python_deps
        fi
    fi
fi

# Make scripts executable
chmod +x "$(dirname "$0")/transpose.py" 2>/dev/null || true
chmod +x "$(dirname "$0")/music_studio.py" 2>/dev/null || true

check_gpu

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE!                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "QUICK START - Karaoke Mode:"
echo ""
echo "  # Create karaoke track from YouTube:"
echo "  python3 music_studio.py karaoke 'https://youtube.com/watch?v=...'"
echo ""
echo "  # Create karaoke from local file:"
echo "  python3 music_studio.py karaoke song.mp3"
echo ""
echo "  # Karaoke + transpose up 2 semitones:"
echo "  python3 music_studio.py karaoke song.mp3 -s 2"
echo ""
echo "OTHER COMMANDS:"
echo ""
echo "  # Just remove vocals (create instrumental):"
echo "  python3 music_studio.py vocals song.mp3"
echo ""
echo "  # Extract only the vocals (acapella):"
echo "  python3 music_studio.py isolate song.mp3"
echo ""
echo "  # Separate into all 4 stems (drums, bass, vocals, other):"
echo "  python3 music_studio.py stems song.mp3"
echo ""
echo "  # Transpose (pitch shift) audio:"
echo "  python3 music_studio.py transpose song.mp3 -s 3"
echo ""
echo "  # Just download from YouTube:"
echo "  python3 music_studio.py download 'https://youtube.com/watch?v=...'"
echo ""
echo "TIP: Create aliases for easier access:"
echo "  echo 'alias karaoke=\"python3 $(dirname "$0")/music_studio.py karaoke\"' >> ~/.zshrc"
echo "  echo 'alias music-studio=\"python3 $(dirname "$0")/music_studio.py\"' >> ~/.zshrc"
echo "  source ~/.zshrc"
echo ""
echo "NOTE: First run may take longer as it downloads the AI model (~1GB)"
echo ""
