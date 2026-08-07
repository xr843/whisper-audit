"""音频预处理与音量测量。"""
import math
import os
import subprocess
import sys
import wave

from . import log


def ensure_cuda_libs():
    """CTranslate2 要在运行时找到 cuBLAS/cuDNN，它们随 torch 的 pip 包装在 nvidia/ 下。
    LD_LIBRARY_PATH 必须在进程启动前生效，所以这里设好后重启自己。

    只能从 main() 调用，绝不能放在模块顶层：import 本模块的进程会被 execv 顶替掉，
    实测这会让 pytest 静默退出（退出码 1、零输出）。"""
    try:
        import nvidia
    except ImportError:
        return
    p = os.path.dirname(nvidia.__file__)
    libs = [os.path.join(p, d, "lib") for d in os.listdir(p)
            if os.path.isdir(os.path.join(p, d, "lib"))]
    if not libs:
        return
    want = ":".join(libs)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if want.split(":")[0] not in cur:
        os.environ["LD_LIBRARY_PATH"] = want + (":" + cur if cur else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def prepare_audio(src, workdir):
    """Whisper 只吃 16kHz 单声道。"""
    wav = os.path.join(workdir, "audio16k.wav")
    if os.path.exists(wav):
        log(f"复用已有 {wav}")
        return wav
    log(f"转码 {os.path.basename(src)} -> 16kHz 单声道")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
    return wav


class Loudness:
    """按时间区间测 RMS 音量。判断空洞是真静音还是漏转，全靠它。"""

    def __init__(self, wav):
        self.w = wave.open(wav, "rb")
        self.sr = self.w.getframerate()
        self.n = self.w.getnframes()

    def db(self, t0, t1):
        import numpy as np
        a0 = max(0, int(t0 * self.sr))
        a1 = min(self.n, int(t1 * self.sr))
        if a1 - a0 < self.sr // 20:
            return float("nan")
        self.w.setpos(a0)
        a = np.frombuffer(self.w.readframes(a1 - a0), dtype=np.int16).astype("float32")
        if a.size == 0:
            return float("nan")
        return 20 * math.log10(max(float((a ** 2).mean()) ** 0.5, 1e-9) / 32768)

    def close(self):
        self.w.close()
