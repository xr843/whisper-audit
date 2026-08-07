"""金标生成与评测的回归测试。

金标是唯一需要人力的环节，所以「人工编辑后还能读回来」这条必须锁死——
读取一旦挑剔，人改完保存却报错，整条链路就废了。
"""
import pytest

from whisper_audit import goldset as G

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
    assert rows[0] == (1.0, 3.0, "上课的时间差不多到了")


def test_make_goldset_respects_time_window(srt_file):
    rows = G.make_goldset(srt_file, start=0.0, end=10.0)
    assert len(rows) == 2


def test_tsv_roundtrip(tmp_path):
    rows = [(1.0, 3.0, "甲乙丙"), (3.5, 6.0, "丁戊己")]
    p = tmp_path / "g.tsv"
    G.write_tsv(rows, str(p))
    assert G.read_tsv(str(p)) == rows


def test_tsv_tolerates_hand_edited_whitespace(tmp_path):
    """人工改完可能留下多余空格或空行，读取必须容忍。"""
    p = tmp_path / "g.tsv"
    p.write_text("1.0\t3.0\t 甲乙丙 \n\n3.5\t6.0\t丁戊己\n", encoding="utf-8")
    assert G.read_tsv(str(p)) == [(1.0, 3.0, "甲乙丙"), (3.5, 6.0, "丁戊己")]


def test_read_tsv_accepts_the_old_two_column_format(tmp_path):
    """旧格式只有起始时间，结束时间退化为起始时间——读得进来，不报错。"""
    p = tmp_path / "old.tsv"
    p.write_text("1.0\t甲乙丙\n", encoding="utf-8")
    assert G.read_tsv(str(p)) == [(1.0, 1.0, "甲乙丙")]


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
    rows[0] = (rows[0][0], rows[0][1], rows[0][2].replace("上课", "上刻"))
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
    assert "这一条在时间窗之外" in "".join(t for _, _, t in G.read_tsv(str(gold)))
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


def test_evaluate_srt_survives_resegmentation(tmp_path):
    """跨版本对比是金标存在的理由：v2 重新切分但一字不差，必须判 0 错。

    曾经按「start 落在窗口内」选取 hyp，v2 里 start 晚于金标末条 start 的
    那一条会被整条丢掉，一字不差的稿子被判出 7 个删除错、CER 35%。
    """
    v1 = tmp_path / "v1.srt"
    v1.write_text("1\n00:00:01,000 --> 00:00:05,000\n前面这句话\n\n"
                  "2\n00:00:06,500 --> 00:00:12,000\n今天要讲的内容是所得税优惠政策\n",
                  encoding="utf-8")
    v2 = tmp_path / "v2.srt"
    v2.write_text("1\n00:00:01,000 --> 00:00:05,000\n前面这句话\n\n"
                  "2\n00:00:06,500 --> 00:00:08,000\n今天要讲的内容是\n\n"
                  "3\n00:00:08,200 --> 00:00:12,000\n所得税优惠政策\n",
                  encoding="utf-8")
    gold = tmp_path / "g.tsv"
    G.write_tsv(G.make_goldset(str(v1)), str(gold))
    assert G.evaluate_srt(str(gold), str(v1))["cer"] == 0.0
    rep = G.evaluate_srt(str(gold), str(v2))
    assert rep["cer"] == 0.0, f"重新切分不该产生错误：{rep}"
    assert rep["dele"] == 0
