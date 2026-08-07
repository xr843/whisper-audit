"""出稿带说话人标签。

铁律：说话人分离只加标注，**一个字的正文都不许动**。这份测试就是那道锁。
"""
import re

from whisperaudit.render import render, speaker_labels


def _rows(spec):
    """spec: [(start, end, text, speaker)] → rows"""
    return [{"start": a, "end": b, "text": t, "speaker": s, "words": []}
            for a, b, t, s in spec]


def _out(tmp_path, rows, dur=60.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    render(rows, dur, str(tmp_path), "测试", {}, "")
    return {p.name: p.read_text(encoding="utf-8") for p in tmp_path.iterdir()}


# ------------------------------------------------------- speaker_labels 纯函数

def test_single_speaker_gets_no_labels():
    """独白录音满屏「S1」是噪声不是信息——一个标签都不该有。"""
    assert speaker_labels(["S1", "S1", "S1"]) == ["", "", ""]


def test_all_unknown_gets_no_labels():
    assert speaker_labels([None, None]) == ["", ""]


def test_label_appears_only_on_change():
    assert speaker_labels(["S1", "S1", "S2", "S2", "S1"]) == ["S1", "", "S2", "", "S1"]


def test_unknown_breaks_continuity():
    """中间那段不知道是谁说的，就不能假装 S1 一直在说——其后要重新标。"""
    assert speaker_labels(["S1", None, "S1", "S2"]) == ["S1", "", "S1", "S2"]


def test_one_known_speaker_plus_unknowns_still_gets_no_labels():
    """只认出一个人、其余没定——仍然不标。

    「S1」在这里不传达任何区分信息（没有 S2 可对比），只会让读者以为
    未标的那些段是另一个人说的。少标不会误导，多标会。
    """
    assert speaker_labels(["S1", None, "S1"]) == ["", "", ""]


# ------------------------------------------------------------------ 出稿

def test_speaker_change_forces_paragraph_break(tmp_path):
    """两个人的话粘进同一段 = 伪造发言归属，比不分段严重得多。

    这两条 row 时间紧挨（间隔 0.1 秒）、字数也远不到 220，
    只有「换人」这一个理由能把它们拆开。
    """
    rows = _rows([(0.0, 3.0, "今天我们讨论第一个议题", "S1"),
                  (3.1, 6.0, "我补充一点不同看法", "S2")])
    md = _out(tmp_path, rows)["测试_全文转录.md"]
    body = [l for l in md.splitlines() if l.startswith("**[")]
    assert len(body) == 2, f"换人没断段：{body}"
    assert "S1" in body[0] and "S2" in body[1]


def test_labels_never_alter_transcript_text(tmp_path):
    """核心不变量：把 speaker 全部抹掉重跑，正文必须一模一样。"""
    spec = [(0.0, 3.0, "今天我们讨论第一个议题", "S1"),
            (3.1, 6.0, "我补充一点不同看法", "S2"),
            (6.2, 9.0, "这个提法我同意", "S1")]
    with_spk = _out(tmp_path / "a", _rows(spec))
    (tmp_path / "b").mkdir()
    without = _out(tmp_path / "b",
                   _rows([(a, b, t, None) for a, b, t, _ in spec]))

    for name in ("测试_全文转录.md", "测试_字幕.srt"):
        # 只摘出汉字，去掉标签/时间戳/序号后比对
        strip = lambda s: re.sub(r"[^一-鿿]", "", s)          # noqa: E731
        assert strip(with_spk[name]) == strip(without[name]), f"{name} 正文被改了"


def test_srt_labels_every_cue_when_multi_speaker(tmp_path):
    """字幕一次只显示一条，观众看不到上一条——「换人才标」在这里没有意义。"""
    rows = _rows([(0.0, 3.0, "今天我们讨论第一个议题", "S1"),
                  (3.1, 6.0, "我补充一点不同看法", "S2"),
                  (6.2, 9.0, "这个提法我同意", "S1")])
    srt = _out(tmp_path, rows)["测试_字幕.srt"]
    assert srt.count("[S1]") == 2 and srt.count("[S2]") == 1


def test_srt_unlabeled_when_single_speaker(tmp_path):
    rows = _rows([(0.0, 3.0, "今天我们讨论第一个议题", "S1"),
                  (3.1, 6.0, "接着说第二点", "S1")])
    assert "[S1]" not in _out(tmp_path, rows)["测试_字幕.srt"]


def test_rows_without_speaker_key_render_unchanged(tmp_path):
    """没跑分离的老路径：rows 里根本没有 speaker 键，不许崩、不许多出标签。"""
    rows = [{"start": 0.0, "end": 3.0, "text": "今天我们讨论第一个议题", "words": []}]
    out = _out(tmp_path, rows)
    assert "S1" not in out["测试_全文转录.md"]
    assert "[S" not in out["测试_字幕.srt"]
