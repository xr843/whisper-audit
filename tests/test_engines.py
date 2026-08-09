"""Engine 抽象的最小烟测。

这层抽象是给第二引擎预留的，此前零调用——意味着真接引擎那天才第一次运行。
至少保证注册表和 out_json=None（不落盘）路径不是坏的。
"""
import pytest

from whisper_audit.engines import Engine, get_engine, register
from whisper_audit.engines.whisper import WhisperEngine


def test_whisper_engine_is_registered():
    e = get_engine("whisper")
    assert isinstance(e, WhisperEngine)
    assert isinstance(e, Engine)
    assert e.name == "whisper"


def test_unknown_engine_raises_with_the_known_list():
    with pytest.raises(KeyError) as ei:
        get_engine("不存在的引擎")
    assert "whisper" in str(ei.value)


def test_engine_kwargs_are_stored():
    e = get_engine("whisper", model_name="tiny", device="cpu", compute="int8")
    assert (e.model_name, e.device, e.compute) == ("tiny", "cpu", "int8")


def test_out_json_none_does_not_crash_the_cache_check():
    """引擎接口允许不落盘；os.path.exists(None) 会抛 TypeError，
    曾经这条路一走就崩——因为从来没人走过。"""
    import inspect

    from whisper_audit.engines import whisper as W
    src = inspect.getsource(W.transcribe_pass)
    assert "out_json and os.path.exists" in src
    src2 = inspect.getsource(W.repatch)
    assert "out_json and os.path.exists" in src2


def test_base_engine_transcribe_is_abstract():
    with pytest.raises(NotImplementedError):
        Engine().transcribe("x.wav")


def test_funasr_engine_is_lazily_importable():
    """funasr 引擎模块按需加载——get_engine 未见注册时先尝试延迟导入。"""
    e = get_engine("funasr")
    assert e.name == "funasr"


def test_funasr_default_model_dir_is_resolved_lazily_via_ensure_model():
    """构造零成本（不猜路径、不摸网络）；默认模型目录在首次转录时经
    ensure_model 解析。此前构造期就把写死的缓存路径塞给 AutoModel。"""
    import inspect

    from whisper_audit.engines import funasr as F
    e = get_engine("funasr")
    assert e.model_dir is None
    assert "ensure_model" in inspect.getsource(F.FunASREngine.transcribe)


def test_ensure_model_returns_cached_dir_without_downloading(tmp_path, monkeypatch):
    """目录已在且文件齐：直接返回，绝不触发下载——离线机器不能被逼联网。"""
    from whisper_audit.engines import ensure_model
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / ".cache/modelscope/hub/models/iic/某模型"
    d.mkdir(parents=True)
    (d / "model.pt").write_text("x")

    def boom(_):
        raise AssertionError("目录齐全时不该下载")
    assert ensure_model("iic/某模型", require=("model.pt",), _download=boom) == str(d)


def test_ensure_model_downloads_when_missing(tmp_path, monkeypatch):
    """目录不在：走 ModelScope 正式 API 下载并返回其落点。

    0.4.1 及之前没有这层——不存在的路径被直接塞给 AutoModel，funasr 拿它
    当 repo id 下载（Invalid repo_id）后 not registered 崩掉；除开发机外
    的一切机器上 funasr 与 --diarize 都起不来（2026-08-09 新机器实测）。"""
    from whisper_audit.engines import ensure_model
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []

    def fake_dl(mid):
        calls.append(mid)
        return "/downloaded/here"
    assert ensure_model("iic/新模型", _download=fake_dl) == "/downloaded/here"
    assert calls == ["iic/新模型"]


def test_ensure_model_redownloads_on_missing_required_file(tmp_path, monkeypatch):
    """目录在但缺 require 文件（seaco 曾缺 seg_dict）：视同没装，重新下载。"""
    from whisper_audit.engines import ensure_model
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / ".cache/modelscope/hub/models/iic/残缺模型"
    d.mkdir(parents=True)
    assert ensure_model("iic/残缺模型", require=("seg_dict",),
                        _download=lambda mid: "/re/downloaded") == "/re/downloaded"
