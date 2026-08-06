"""拼音纠错接入流水线的回归测试。

重点不在「能不能改」，而在「改了有没有如实记账」——
凡是自动改正文的逻辑，改动必须可追溯，否则出问题时无从查起。
"""
import pytest

from audio_transcribe import merge as M

TERMS = {"terms": ["财税管理"], "fixes": []}


def one_pass(text, start=0.0, end=10.0, dur=60.0):
    return [{"duration": dur,
             "segments": [{"start": start, "end": end, "text": text,
                           "avg_logprob": -0.3, "no_speech_prob": 0.1}]}]


def test_combine_applies_pinyin_fix_and_records_hits():
    rows, hits = M.combine(one_pass("才税管理的要点在这里说清楚"), [], TERMS, [], 60.0,
                           pinyin=True, return_hits=True)
    assert "财税管理" in "".join(r["text"] for r in rows)
    assert hits and hits[0]["to"] == "财税管理"
    assert "t" in hits[0], "改动必须带时间才追溯得到"


def test_pinyin_fix_can_be_disabled():
    rows, hits = M.combine(one_pass("才税管理的要点在这里说清楚"), [], TERMS, [], 60.0,
                           pinyin=False, return_hits=True)
    assert "才税管理" in "".join(r["text"] for r in rows)
    assert hits == []


def test_combine_without_return_hits_still_returns_plain_rows():
    """老调用点不传 return_hits，必须还是拿到一个 list 而不是 tuple。"""
    rows = M.combine(one_pass("才税管理的要点在这里说清楚"), [], TERMS, [], 60.0)
    assert isinstance(rows, list)
    assert rows and isinstance(rows[0], dict)


def test_hits_are_not_double_counted():
    """norm() 带副作用，每条 segment 只能调一次；调两次记账会翻倍。"""
    _, hits = M.combine(one_pass("才税管理的要点在这里说清楚"), [], TERMS, [], 60.0,
                        pinyin=True, return_hits=True)
    assert len(hits) == 1, f"同一处改动被记了 {len(hits)} 次"


def test_losing_patch_candidates_do_not_pollute_the_ledger():
    """补转会为同一区段生成多组候选，落选的那组不能进记账清单——
    否则质检报告会列出根本没进成品的修改。"""
    passes = one_pass("这里原本很短", 0.0, 30.0, 60.0)
    patch = [{"span": [0.0, 30.0], "label": "段内饥饿", "attempts": [
        {"mode": "novad", "rows": [
            {"start": 0.0, "end": 30.0, "text": "才税管理" * 2 + "补转捞回来的完整内容长得多"}]},
        {"mode": "vad_loose", "rows": [
            {"start": 0.0, "end": 30.0, "text": "才税管理"}]},
    ]}]
    rows, hits = M.combine(passes, patch, TERMS, [], 60.0, pinyin=True, return_hits=True)
    joined = "".join(r["text"] for r in rows)
    assert "补转捞回来的完整内容" in joined, "字多的那组候选应当胜出"
    # 落选那组也含「才税管理」，若被计入则会多出记账条目
    assert len(hits) == joined.count("财税管理"), \
        f"记账 {len(hits)} 条，成品里只有 {joined.count('财税管理')} 处"


def test_terms_without_terms_field_is_a_noop():
    rows, hits = M.combine(one_pass("这段文本没有任何术语需要修正"), [],
                           {"fixes": []}, [], 60.0, return_hits=True)
    assert hits == []
    assert "这段文本没有任何术语需要修正" in "".join(r["text"] for r in rows)


def test_loose_is_off_by_default():
    """近音归并碰撞面大，必须显式开启。

    「参说」loose 下与「场所」同键，严格模式下不该被改。
    """
    terms = {"terms": ["大型活动场所"], "fixes": []}
    text = "这里提到大型活动参说的登记问题需要注意"
    strict, h1 = M.combine(one_pass(text), [], terms, [], 60.0, pinyin=True,
                           return_hits=True)
    loose, h2 = M.combine(one_pass(text), [], terms, [], 60.0, pinyin=True, loose=True,
                          return_hits=True)
    assert "参说" in "".join(r["text"] for r in strict) and h1 == []
    assert "大型活动场所" in "".join(r["text"] for r in loose) and h2


def test_pinyin_fix_is_off_by_default():
    """默认必须关。无调拼音会把两个真实存在、含义不同的词判成同一个：

        节余分配 → 结余分配   （jie yu，会计上是两个不同概念）
        内部空值 → 内部控制   （kong zhi）

    这不是「补长词就能堵住的词边界问题」，是拼音粒度本身的极限。
    """
    rows, hits = M.combine(one_pass("才税管理的要点在这里说清楚"), [], TERMS, [], 60.0,
                           return_hits=True)
    assert "才税管理" in "".join(r["text"] for r in rows)
    assert hits == []


@pytest.mark.xfail(
    strict=True,
    reason="已知缺陷：无调拼音把「节余/结余」「空值/控制」这类真实存在、含义不同的词"
           "判成同一个。修好（如引入分词/上下文校验）后此测试转 XPASS 变红，"
           "提醒同步更新文档与默认开关。")
def test_real_words_survive_pinyin_fix_when_enabled():
    """开启拼音纠错后，真实存在的词不该被改成同音的另一个词——
    这是**期望行为**；当前实现做不到，故标 xfail 让欠债在 CI 输出里可见。"""
    from audio_transcribe.terms import pinyin_fix
    terms = {"terms": ["结余分配", "内部控制"]}
    for src in ("本年度完成节余分配后转入下年度", "那条内部空值字段没有清理干净"):
        out, hits = pinyin_fix(src, terms)
        assert out == src
        assert hits == []


def test_hits_are_revalidated_against_the_final_text():
    """记账必须按最终文本复核。

    merge_rows 会用 strip_common 裁掉重复字块，被纠错的词可能正好在裁掉的那段。
    实测两路高度重叠时，1 处真实改动曾记出 2 条，第二条指向一个压根没有该词的行——
    审计人照着时间戳去回听会什么都找不到。
    """
    passes = [{"duration": 60.0, "segments": [
        {"start": 0.0, "end": 10.0, "text": "才税管理的要点在这里说清楚了各位注意听讲",
         "avg_logprob": -0.3, "no_speech_prob": 0.1},
        {"start": 0.2, "end": 10.5,
         "text": "才税管理的要点在这里说清楚了各位注意听讲另外补一句全新的话",
         "avg_logprob": -0.3, "no_speech_prob": 0.1}]}]
    rows, hits = M.combine(passes, [], TERMS, [], 60.0, pinyin=True, return_hits=True)
    joined = "".join(r["text"] for r in rows)
    assert len(hits) == joined.count("财税管理"), \
        f"记账 {len(hits)} 条，成品里只有 {joined.count('财税管理')} 处：{hits}"
    for h in hits:
        assert "pos" not in h, "pos 是裁剪前的偏移，裁完就失效，留着比没有更误导"
        assert "t" in h
