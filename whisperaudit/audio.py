"""音频预处理与音量测量。"""
import math
import os
import subprocess
import sys
import wave

from . import log


def nvidia_lib_dirs():
    """pip 装的 NVIDIA 运行时库目录（nvidia/*/lib）。找不到返回空列表。

    `nvidia` 是 PEP 420 **隐式命名空间包**——它没有 `__init__.py`，因此
    `nvidia.__file__` 是 `None`，只有 `__path__` 可用。

    2026-08-07 冷启动实测踩到：在只装了 `nvidia-cublas-cu12` +
    `nvidia-cudnn-cu12`（不装 torch）的干净环境里，原先的
    `os.path.dirname(nvidia.__file__)` 直接抛
    `TypeError: expected str, bytes or os.PathLike object, not NoneType`，
    整个 run 崩在第一行。开发机上一直没暴露，纯粹因为那里装了 torch。
    **而「装 CUDA 库但不装 torch」正是本项目文档推荐的路径。**

    命名空间包的 `__path__` 还可能有多个根（不同 site-packages），一并扫。
    """
    try:
        import nvidia
    except ImportError:
        return []
    roots = list(getattr(nvidia, "__path__", None) or [])
    if not roots and getattr(nvidia, "__file__", None):
        roots = [os.path.dirname(nvidia.__file__)]
    out = []
    for r in roots:
        try:
            names = sorted(os.listdir(r))
        except OSError:
            continue
        out.extend(os.path.join(r, d, "lib") for d in names
                   if os.path.isdir(os.path.join(r, d, "lib")))
    return out


def restart_argv(main_module=None, argv=None, executable=None):
    """重启自己时该用的 argv。纯函数，可测。

    **`-m` 启动必须保持 `-m` 形式重启。** `python -m whisperaudit.cli` 下
    `sys.argv[0]` 是 `cli.py` 的文件路径，直接 `[executable] + argv` 会把重启
    变成「以脚本方式运行 cli.py」——而 `cli.py` 里没有 `if __name__ == "__main__"`，
    于是它定义完所有函数就**正常退出，退出码 0、零输出、一件事没做**。

    2026-08-07 实测（清空 LD_LIBRARY_PATH 后）：
        python3 -m whisperaudit.cli run 录音.wav   → 退出码 0，无输出，无产物
        python3 transcribe.py 录音.wav             → 正常报错

    这是本项目最忌讳的那类故障：看起来像成功。

    判据用 `__main__.__spec__`——`-m` 启动时它不是 None 且 `.name` 就是模块名；
    脚本启动（`transcribe.py`）和 console script（装包后的 `whisperaudit`）
    都是 None，走原路径。
    """
    import sys as _sys
    if main_module is None:
        import __main__ as main_module
    argv = list(_sys.argv if argv is None else argv)
    executable = executable or _sys.executable
    spec = getattr(main_module, "__spec__", None)
    name = getattr(spec, "name", None)
    if name:
        return [executable, "-m", name] + argv[1:]
    return [executable] + argv


def ensure_cuda_libs():
    """CTranslate2 要在运行时找到 cuBLAS/cuDNN，它们是独立的 nvidia-*-cu12 pip 包。
    LD_LIBRARY_PATH 必须在进程启动前生效，所以这里设好后重启自己。

    只能从 main() 调用，绝不能放在模块顶层：import 本模块的进程会被 execv 顶替掉，
    实测这会让 pytest 静默退出（退出码 1、零输出）。"""
    libs = nvidia_lib_dirs()
    if not libs:
        return
    want = ":".join(libs)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if want.split(":")[0] not in cur:
        os.environ["LD_LIBRARY_PATH"] = want + (":" + cur if cur else "")
        os.execv(sys.executable, restart_argv())


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
