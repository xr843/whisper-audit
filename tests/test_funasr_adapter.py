"""FunASR 适配器测试。夹具形状来自 2026-08-06 真机探测，不是照文档推测。

实测格式：text 字间带空格，timestamp 逐字 [start_ms, end_ms] 与字一一对应。
"""
from whisper_audit.engines.funasr import to_segments


def test_real_shape_char_level_timestamps_become_words():
    raw = [{"key": "x", "text": "转 录 文 本",
            "timestamp": [[53150, 53330], [53330, 53550], [53550, 53770], [53770, 54000]]}]
    segs = to_segments(raw)
    assert len(segs) == 1
    s = segs[0]
    assert s["text"] == "转录文本"
    assert s["start"] == 53.15 and s["end"] == 54.0
    assert [w["word"] for w in s["words"]] == ["转", "录", "文", "本"]
    assert s["words"][0]["start"] == 53.15


def test_long_gap_splits_segments():
    raw = [{"key": "x", "text": "甲 乙 丙 丁",
            "timestamp": [[0, 200], [200, 400], [5000, 5200], [5200, 5400]]}]
    segs = to_segments(raw, gap_ms=800)
    assert [s["text"] for s in segs] == ["甲乙", "丙丁"]
    assert segs[1]["start"] == 5.0


def test_overlong_run_is_split_at_max_duration():
    """慢速歌唱连绵不断时不许整篇一段——下游按 30 秒桶合并，超长段会跨桶捣乱。"""
    n = 300
    raw = [{"key": "x", "text": " ".join("字" for _ in range(n)),
            "timestamp": [[i * 200, i * 200 + 180] for i in range(n)]}]   # 60 秒连读
    segs = to_segments(raw, max_dur_s=28.0)
    assert len(segs) >= 2
    assert all(s["end"] - s["start"] <= 28.5 for s in segs)
    assert "".join(s["text"] for s in segs) == "字" * n


def test_mismatched_timestamp_degrades_not_crashes():
    """text 与 timestamp 对不上时整条降级（words 置空），下游自动退化，不许崩。"""
    raw = [{"key": "x", "text": "甲 乙 丙", "timestamp": [[0, 200]]}]
    segs = to_segments(raw)
    assert len(segs) == 1
    assert segs[0]["text"] == "甲乙丙"
    assert segs[0]["words"] == []


def test_confidence_defaults_do_not_trip_hallucination_filter():
    """FunASR 不给置信度。默认值必须是高置信（0.0），
    否则会被幻觉判据 avg_logprob < -1.0 整段误杀。"""
    raw = [{"key": "x", "text": "甲", "timestamp": [[0, 200]]}]
    s = to_segments(raw)[0]
    assert s["avg_logprob"] >= -1.0
    assert s["no_speech_prob"] < 0.6


def test_empty_and_missing_fields_are_tolerated():
    assert to_segments([{"key": "x", "text": "", "timestamp": []}]) == []
    assert to_segments([{"key": "x"}]) == []


def test_duration_survives_float32_wav(tmp_path):
    """FLEURS 等公开集是 float32 WAV（format 3），标准库 wave 直接抛错。
    实测在跑 FLEURS 评测时撞出：适配器读时长那一步崩掉整个评测。"""
    import numpy as np
    import soundfile

    from whisper_audit.engines.funasr import _duration
    p = tmp_path / "f32.wav"
    soundfile.write(str(p), np.zeros(16000, dtype="float32"), 16000, subtype="FLOAT")
    d = _duration(str(p), [])
    assert abs(d - 1.0) < 0.01


def test_duration_falls_back_to_last_segment_end(tmp_path):
    from whisper_audit.engines.funasr import _duration
    d = _duration(str(tmp_path / "不存在.wav"), [{"end": 12.5}])
    assert d == 12.5


# ------------------------------------------------- 长音频分窗（OOM 修复）

def _audio(sr, spec):
    """spec: [(秒数, 幅度)] → float32 波形。"""
    import numpy as np
    return np.concatenate([np.full(int(d * sr), amp, dtype="float32")
                           for d, amp in spec])


def test_short_audio_is_not_split():
    """短音频必须原样单窗——分窗只为救 OOM，不许改变已验证的短音频路径。"""
    import numpy as np
    from whisper_audit.engines.funasr import split_points
    a = np.ones(16000 * 60, dtype="float32")
    assert split_points(a, 16000, max_chunk_s=300.0) == [0, len(a)]


def test_long_audio_boundaries_land_in_silence():
    """名义边界附近有停顿时，切点必须挪进停顿里——不许把字从中间切开。

    2026-08-07 实测背景：Paraformer 注意力显存随长度平方增长，
    15 分钟能跑、34 分钟要 33GiB 直接 OOM。而 README 推荐它转的
    正是长音频——三个困难域评测同时崩掉才暴露这一点。
    """
    from whisper_audit.engines.funasr import split_points
    sr = 1000
    # 25s 音频：9.5~10.5s 与 19~20s 是静音，其余响
    a = _audio(sr, [(9.5, 0.5), (1.0, 0.0), (8.5, 0.5), (1.0, 0.0), (5.0, 0.5)])
    pts = split_points(a, sr, max_chunk_s=10.0, search_s=2.0)
    inner = [p / sr for p in pts[1:-1]]
    assert len(inner) >= 2
    assert any(9.5 <= t <= 10.5 for t in inner), f"第一刀没落进停顿：{inner}"
    assert any(19.0 <= t <= 20.0 for t in inner), f"第二刀没落进停顿：{inner}"


def test_split_covers_everything_without_overlap():
    """切点严格递增、首尾覆盖全长——丢样本或重复样本都是静默事故。"""
    from whisper_audit.engines.funasr import split_points
    sr = 1000
    a = _audio(sr, [(35.0, 0.3)])          # 全程响，无停顿可找
    pts = split_points(a, sr, max_chunk_s=10.0, search_s=2.0)
    assert pts[0] == 0 and pts[-1] == len(a)
    assert all(b > a_ for a_, b in zip(pts, pts[1:])), f"切点必须严格递增：{pts}"


def test_no_silence_still_splits_near_nominal():
    """整段都响时退化为按名义位置切——错一两个字远好于 OOM 崩整条。"""
    from whisper_audit.engines.funasr import split_points
    sr = 1000
    a = _audio(sr, [(35.0, 0.3)])
    pts = split_points(a, sr, max_chunk_s=10.0, search_s=2.0)
    for a_, b in zip(pts, pts[1:]):
        assert (b - a_) / sr <= 10.0 + 2.0 + 0.3, "单窗不许超过名义长度+搜索半径"
