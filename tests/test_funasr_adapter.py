"""FunASR 适配器测试。夹具形状来自 2026-08-06 真机探测，不是照文档推测。

实测格式：text 字间带空格，timestamp 逐字 [start_ms, end_ms] 与字一一对应。
"""
from audio_transcribe.engines.funasr import to_segments


def test_real_shape_char_level_timestamps_become_words():
    raw = [{"key": "x", "text": "菩 萨 相 品",
            "timestamp": [[53150, 53330], [53330, 53550], [53550, 53770], [53770, 54000]]}]
    segs = to_segments(raw)
    assert len(segs) == 1
    s = segs[0]
    assert s["text"] == "××相品"
    assert s["start"] == 53.15 and s["end"] == 54.0
    assert [w["word"] for w in s["words"]] == ["菩", "萨", "相", "品"]
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
