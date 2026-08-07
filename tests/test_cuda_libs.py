"""CUDA 运行时库定位。

这条路径 2026-08-07 冷启动实测崩过：干净环境里 `pip install whisperaudit[whisper]`
+ CUDA 库、不装 torch，`nvidia.__file__` 是 None，原实现直接 TypeError，
整个 run 在第一行就死。开发机因为装了 torch 一直没暴露。

`ensure_cuda_libs()` 本身会 os.execv 自己，测不了；把纯粹的目录发现逻辑
拆成 `nvidia_lib_dirs()` 就能测——这也是当初该拆的边界。
"""
import sys
import types

import pytest

from whisperaudit.audio import nvidia_lib_dirs


@pytest.fixture
def fake_nvidia(monkeypatch):
    """往 sys.modules 塞一个假的 nvidia 包，测完自动摘掉。"""
    def install(**attrs):
        mod = types.ModuleType("nvidia")
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, "nvidia", mod)
        return mod
    return install


def _make_tree(root, names):
    for n in names:
        (root / n / "lib").mkdir(parents=True)
    return root


def test_namespace_package_without_file_attr(tmp_path, fake_nvidia):
    """核心回归：命名空间包的 __file__ 是 None，只有 __path__ 可用。

    实测崩溃信息：
        TypeError: expected str, bytes or os.PathLike object, not NoneType
    """
    _make_tree(tmp_path, ["cublas", "cudnn"])
    fake_nvidia(__file__=None, __path__=[str(tmp_path)])

    dirs = nvidia_lib_dirs()
    assert len(dirs) == 2
    assert all(d.endswith("/lib") for d in dirs)
    assert any("cublas" in d for d in dirs)


def test_regular_package_with_file_attr(tmp_path, fake_nvidia):
    """torch 在场时 nvidia 可能是常规包——老路径不能因为修复而失效。"""
    _make_tree(tmp_path, ["cublas"])
    fake_nvidia(__file__=str(tmp_path / "__init__.py"), __path__=[])

    assert [d for d in nvidia_lib_dirs() if "cublas" in d]


def test_multiple_namespace_roots(tmp_path, fake_nvidia):
    """命名空间包可以横跨多个 site-packages，每个根都要扫。"""
    a = _make_tree(tmp_path / "a", ["cublas"])
    b = _make_tree(tmp_path / "b", ["cudnn"])
    fake_nvidia(__file__=None, __path__=[str(a), str(b)])

    dirs = nvidia_lib_dirs()
    assert any("cublas" in d for d in dirs) and any("cudnn" in d for d in dirs)


def test_missing_nvidia_package_is_not_an_error(monkeypatch):
    """纯 CPU 用户根本没有 nvidia 包——返回空列表，绝不能抛。"""
    monkeypatch.setitem(sys.modules, "nvidia", None)   # import 得到 None → ImportError
    assert nvidia_lib_dirs() == []


def test_root_without_lib_subdirs(tmp_path, fake_nvidia):
    """有 nvidia 包但没有任何 */lib——不能把无关目录当成库路径。"""
    (tmp_path / "cuda_runtime" / "include").mkdir(parents=True)
    fake_nvidia(__file__=None, __path__=[str(tmp_path)])
    assert nvidia_lib_dirs() == []


def test_unreadable_root_is_skipped(tmp_path, fake_nvidia):
    """__path__ 里有不存在的目录（卸载残留）时跳过，不能崩。"""
    good = _make_tree(tmp_path / "good", ["cublas"])
    fake_nvidia(__file__=None, __path__=[str(tmp_path / "gone"), str(good)])
    assert len(nvidia_lib_dirs()) == 1


# ------------------------------------------------------------------ 音量测量

def test_short_interval_returns_nan(tmp_path):
    """短于 50ms 的区间必须返回 NaN，不能返回一个数。

    2026-08-07 变异测试发现这条契约没有任何测试守着。
    它不是内部细节——`audit()` 靠 `if db != db: continue` 认出「这段太短，
    量不出可信音量」。改成返回真实 RMS 的话，短段会拿到一个由极少量样本
    算出的、方差极大的音量值，直接喂给死空气与幻觉判据：
    偏低就误删真内容，偏高就漏掉幻觉。两个方向都是静默错误。
    """
    import math
    import wave

    import numpy as np

    from whisperaudit.audio import Loudness

    wav = tmp_path / "t.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.ones(16000, dtype="int16") * 3000).tobytes())

    loud = Loudness(str(wav))
    try:
        assert math.isnan(loud.db(0.0, 0.02)), "20ms 区间必须是 NaN"
        assert not math.isnan(loud.db(0.0, 0.5)), "500ms 区间必须给出真实音量"
    finally:
        loud.close()


# ------------------------------------------------- 重启自己时的 argv（静默失败）

def test_dash_m_invocation_restarts_as_dash_m():
    """`-m` 启动必须以 `-m` 形式重启，否则整个进程「成功」退出而一事无成。

    2026-08-07 实测（清空 LD_LIBRARY_PATH 后）：
        python3 -m whisperaudit.cli run 录音.wav  → 退出码 0，零输出，无产物
        python3 transcribe.py 录音.wav            → 正常报错

    根因：`-m` 下 sys.argv[0] 是 cli.py 的**文件路径**，
    `[executable] + argv` 把重启变成「以脚本方式跑 cli.py」，
    而 cli.py 当时没有 `if __name__ == "__main__"`，定义完函数就正常退出。

    这是本项目最忌讳的故障类型：**看起来像成功**。
    """
    from whisperaudit.audio import restart_argv

    mod = types.ModuleType("__main__")
    mod.__spec__ = types.SimpleNamespace(name="whisperaudit.cli")

    got = restart_argv(mod, ["/path/to/whisperaudit/cli.py", "run", "a.wav"], "/usr/bin/python3")
    assert got == ["/usr/bin/python3", "-m", "whisperaudit.cli", "run", "a.wav"]


def test_script_invocation_restarts_as_script():
    """脚本启动（transcribe.py）与装包后的 console script：__spec__ 为 None，走原路径。"""
    from whisperaudit.audio import restart_argv

    mod = types.ModuleType("__main__")
    mod.__spec__ = None

    got = restart_argv(mod, ["/usr/local/bin/whisperaudit", "run", "a.wav"], "/usr/bin/python3")
    assert got == ["/usr/bin/python3", "/usr/local/bin/whisperaudit", "run", "a.wav"]


def test_module_without_spec_attribute_at_all():
    """有些嵌入式解释器的 __main__ 连 __spec__ 属性都没有——不能因此崩掉。"""
    from whisperaudit.audio import restart_argv

    mod = types.ModuleType("__main__")
    if hasattr(mod, "__spec__"):
        del mod.__spec__

    assert restart_argv(mod, ["x.py", "run"], "/py") == ["/py", "x.py", "run"]


def test_cli_module_is_runnable_as_a_script():
    """兜底闸门：cli.py 必须有 __main__ 守卫。

    没有它，任何把重启退化成脚本形式的路径都会静默变成 no-op。
    """
    import pathlib

    import whisperaudit.cli as c

    src = pathlib.Path(c.__file__).read_text(encoding="utf-8")
    assert '__name__ == "__main__"' in src, "cli.py 缺 __main__ 守卫"
