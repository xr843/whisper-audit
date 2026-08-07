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
