"""金标生成与评测的回归测试。

金标是唯一需要人力的环节，所以「人工编辑后还能读回来」这条必须锁死——
读取一旦挑剔，人改完保存却报错，整条链路就废了。
"""
import pytest

from audio_transcribe import goldset as G

SRT = """1
00:00:01,000 --> 00:00:03,000
上课的时间差不多到了

2
00:00:03,500 --> 00:00:06,000
请外面的学员进场入座

3
00:10:00,000 --> 00:10:04,000
这一条在时间窗之外
"""


@pytest.fixture
def srt_file(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text(SRT, encoding="utf-8")
    return str(p)


def test_make_goldset_extracts_rows(srt_file):
    rows = G.make_goldset(srt_file)
    assert len(rows) == 3
    assert rows[0] == (1.0, "上课的时间差不多到了")


def test_make_goldset_respects_time_window(srt_file):
    rows = G.make_goldset(srt_file, start=0.0, end=10.0)
    assert len(rows) == 2


def test_tsv_roundtrip(tmp_path):
    rows = [(1.0, "甲乙丙"), (3.5, "丁戊己")]
    p = tmp_path / "g.tsv"
    G.write_tsv(rows, str(p))
    assert G.read_tsv(str(p)) == rows


def test_tsv_tolerates_hand_edited_whitespace(tmp_path):
    """人工改完可能留下多余空格或空行，读取必须容忍。"""
    p = tmp_path / "g.tsv"
    p.write_text("1.0\t 甲乙丙 \n\n3.5\t丁戊己\n", encoding="utf-8")
    assert G.read_tsv(str(p)) == [(1.0, "甲乙丙"), (3.5, "丁戊己")]


def test_parse_hms_accepts_three_forms():
    assert G.parse_hms("00:12:34") == 754.0
    assert G.parse_hms("12:34") == 754.0
    assert G.parse_hms("754") == 754.0
    assert G.parse_hms("") is None


def test_evaluate_srt_scores_zero_on_untouched_goldset(srt_file, tmp_path):
    """金标初稿就是 ASR 输出，没改错字时 CER 必须是 0——否则说明
    时间窗或拼接逻辑本身在制造错误。"""
    gold = tmp_path / "g.tsv"
    G.write_tsv(G.make_goldset(srt_file), str(gold))
    rep = G.evaluate_srt(str(gold), srt_file)
    assert rep["cer"] == 0.0
    assert rep["n_ref"] > 0


def test_evaluate_srt_counts_hand_corrections(srt_file, tmp_path):
    """把金标里一个字改掉，评测要能算出来。"""
    rows = G.make_goldset(srt_file)
    rows[0] = (rows[0][0], rows[0][1].replace("上课", "上刻"))
    gold = tmp_path / "g.tsv"
    G.write_tsv(rows, str(gold))
    rep = G.evaluate_srt(str(gold), srt_file)
    assert rep["sub"] == 1
    assert rep["homo"] == 1          # 课/刻 同音


def test_evaluate_srt_window_covers_last_gold_row(srt_file, tmp_path):
    """上界用 <= 才能把末条含进来；写成 < 会静默漏掉最后一条的内容。"""
    gold = tmp_path / "g.tsv"
    G.write_tsv(G.make_goldset(srt_file), str(gold))
    rep = G.evaluate_srt(str(gold), srt_file)
    assert "这一条在时间窗之外" in "".join(t for _, t in G.read_tsv(str(gold)))
    assert rep["dele"] == 0, "末条被漏掉会表现为一堆删除错误"


def test_evaluate_srt_rejects_empty_goldset(tmp_path):
    p = tmp_path / "empty.tsv"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        G.evaluate_srt(str(p), str(p))


def test_find_srt_errors_when_ambiguous(tmp_path):
    (tmp_path / "a.srt").write_text(SRT, encoding="utf-8")
    (tmp_path / "b.srt").write_text(SRT, encoding="utf-8")
    with pytest.raises(ValueError):
        G.find_srt(str(tmp_path))
