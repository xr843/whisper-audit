#!/usr/bin/env python3
"""长音频转录流水线 —— 兼容入口。实现在 audio_transcribe/ 包里。

    python3 transcribe.py 录音.mp3 -o 输出目录 --profile meeting
"""
import sys

from audio_transcribe.cli import main

if __name__ == "__main__":
    sys.exit(main())
