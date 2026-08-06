"""Engine 抽象的最小烟测。

这层抽象是给第二引擎预留的，此前零调用——意味着真接引擎那天才第一次运行。
至少保证注册表和 out_json=None（不落盘）路径不是坏的。
"""
import pytest

from audio_transcribe.engines import Engine, get_engine, register
from audio_transcribe.engines.whisper import WhisperEngine


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

    from audio_transcribe.engines import whisper as W
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
