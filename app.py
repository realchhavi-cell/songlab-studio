#!/usr/bin/env python3
"""
SongLab - Hugging Face Spaces Entry Point
"""

from songlab_ui import app

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
