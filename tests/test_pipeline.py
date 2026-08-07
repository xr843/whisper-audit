"""无 GPU、无音频的回归测试。

覆盖的是「会静悄悄吃掉内容 / 静悄悄虚报覆盖率」的那几条逻辑——
它们跑起来永远不报错，只会让成品变差，所以只能靠测试盯着。

    python3 -m pytest tests/ -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisperaudit import audit, merge, render


def seg(start, end, text, logprob=-0.3, words=None):
    d = {"start": start, "end": end, "text": text,
         "avg_logprob": logprob, "no_speech_prob": 0.1}
    if words is not None:
        d["words"] = words
    return d


def even_words(start, end, text):
    """把 text 均匀铺在 [start,end] 上，用于造测试数据。"""
    n = len(text)
    step = (end - start) / n
    return [{"start": round(start + i * step, 3),
             "end": round(start + (i + 1) * step, 3), "word": c}
            for i, c in enumerate(text)]


# ---------------------------------------------------------------- 段内饥饿

def test_starved_segment_is_detected_even_with_high_confidence():
    """实测最坏的一处：296 秒只转出 128 字，但 avg_logprob = -0.28。

    旧逻辑的两条判据都写成 `low_conf and ...`，高置信直接豁免，
    于是这 296 秒被算作「已覆盖」，1000 字左右的内容凭空消失且无人知晓。
    """
    segs = [seg(1575.0, 1871.0, "你" * 128, logprob=-0.28)]
    spans = audit.starved_spans(segs)
    assert spans, "高置信度的饥饿段必须被检出"
    assert min(a for a, _, _ in spans) <= 1580
    assert max(b for _, b, _ in spans) >= 1866


def test_starved_span_is_chopped_for_repatch():
    """整段 296 秒丢给补转会退化成一次全量重跑，必须切块。"""
    segs = [seg(1575.0, 1871.0, "你" * 128, logprob=-0.28)]
    spans = audit.starved_spans(segs, max_span=90.0)
    assert len(spans) >= 3
    assert all(b - a <= 90.0 + 1e-6 for a, b, _ in spans)


def test_normal_speech_is_not_starved():
    """全片中位密度 3.63 字/秒，正常讲话不能被误判成饥饿。"""
    segs = [seg(0.0, 30.0, "字" * 100)]
    assert audit.starved_spans(segs) == []


def test_short_segment_is_not_starved():
    """短段交给幻觉那条路处理，不进补转清单，否则清单会被碎片淹没。"""
    segs = [seg(0.0, 6.0, "嗯嗯")]
    assert audit.starved_spans(segs) == []


def test_audit_includes_starved_spans(monkeypatch):
    class FakeLoud:
        def db(self, t0, t1):
            return -28.0

    data = {"duration": 2000.0,
            "segments": [seg(0.0, 1500.0, "字" * 4000),
                         seg(1500.0, 1900.0, "字" * 120, logprob=-0.25)]}
    rep = audit.audit(data, FakeLoud())
    labels = [lab for _, _, lab in rep["spans"]]
    assert any("饥饿" in lab for lab in labels)


# ------------------------------------------------- 死空气：零置信度引擎的幻觉

class _Loud:
    """按时间段返回预设音量。区间用 [start, end) 匹配。"""

    def __init__(self, plan, default=-28.0):
        self.plan, self.default = plan, default

    def db(self, t0, t1):
        for a, b, v in self.plan:
            if a <= t0 < b:
                return v
        return self.default


def test_dead_air_segment_is_dropped():
    """零置信度引擎（funasr/qwen）的 avg_logprob 恒为 0.0。

    加这条判据之前，`if not (by_word or low_conf): continue` 让整个音量侧
    检测对它们是死代码——静音段吐出的「嗯。」「没有没有没有」全都进成品。
    2026-08-07 实测 funasr 在 5 秒纯静音上吐 17 个字。
    """
    data = {"duration": 100.0,
            "segments": [seg(0.0, 60.0, "字" * 200, logprob=0.0),
                         seg(60.0, 65.0, "没有没有没有有有有没有", logprob=0.0)]}
    rep = audit.audit(data, _Loud([(60.0, 65.0, -270.0)]))
    assert [60.0, 65.0] in rep["drop"], f"死空气段没被剔除：{rep['drop']}"
    assert any("没有" in t for _, _, t, _ in rep["hallu"])


def test_quiet_real_speech_is_never_dropped():
    """护栏：轻声讲话不许被当成死空气删掉。

    误删真内容比漏掉一处幻觉严重得多——删除率是本项目的立身指标。
    比中位低 15 dB（真人压低嗓门的量级）必须活下来；阈值 25 dB 就是
    为这个留的余量。
    """
    data = {"duration": 100.0,
            "segments": [seg(0.0, 60.0, "字" * 200, logprob=0.0),
                         seg(60.0, 90.0, "这段是压低嗓门说的但确实是内容", logprob=0.0)]}
    rep = audit.audit(data, _Loud([(60.0, 90.0, -43.0)]))   # 中位 -28，低 15dB
    assert rep["drop"] == [], f"轻声真内容被误删：{rep['drop']}"


def test_empty_segment_in_silence_is_not_flagged():
    """静音处本来就没转出字——没有幻觉可言，不该记账。"""
    data = {"duration": 100.0,
            "segments": [seg(0.0, 60.0, "字" * 200, logprob=0.0),
                         seg(60.0, 65.0, "   ", logprob=0.0)]}
    rep = audit.audit(data, _Loud([(60.0, 65.0, -270.0)]))
    assert rep["drop"] == []


# ---------------------------------------------------------------- 终审

def test_final_audit_reports_effective_speech_not_wall_clock():
    """质检报告过去审的是 pass1，交付的却是合并稿，两个数差了 7 个百分点。"""
    rows = [{"start": 0.0, "end": 100.0, "text": "字" * 300},
            {"start": 100.0, "end": 200.0, "text": "字" * 30}]
    rep = audit.audit_rows(rows, 400.0)
    assert rep["cover_pct"] == pytest.approx(50.0, abs=0.1)
    # 后一段密度 0.3 字/秒，不能算进有效语音
    assert rep["speech_pct"] < rep["cover_pct"]
    assert rep["starved"] == 1


def test_final_audit_flags_nothing_on_clean_rows():
    rows = [{"start": t, "end": t + 10.0, "text": "字" * 35} for t in range(0, 100, 10)]
    rep = audit.audit_rows(rows, 100.0)
    assert rep["cover_pct"] == pytest.approx(100.0, abs=0.1)
    assert rep["starved"] == 0


# ---------------------------------------------------------------- 字幕

def test_srt_cues_never_overlap():
    """实测 19% 的字幕条与前一条时间重叠——两路切分点不同，合并后从不重算时间。"""
    rows = [{"start": 25.0, "end": 108.0, "text": "上课的时间差不多到了"},
            {"start": 65.0, "end": 113.0, "text": "请外面的学员进场入座"},
            {"start": 112.0, "end": 142.0, "text": "我们现在开始上课"}]
    cues = render.resplit_rows(rows)
    for a, b in zip(cues, cues[1:]):
        assert a["end"] <= b["start"] + 1e-6, "字幕条不允许时间重叠"
        assert a["start"] <= b["start"]


def test_srt_splits_overlong_cue_using_word_times():
    """最长一条字幕 296 秒 / 128 字，没法拿来逐句回听核对。"""
    text = "这是一段很长的话需要按词的时间切成若干条字幕方便逐句核对原音"
    rows = [{"start": 0.0, "end": 120.0, "text": text,
             "words": even_words(0.0, 120.0, text)}]
    cues = render.resplit_rows(rows, max_dur=8.0, max_chars=16)
    assert len(cues) >= 4
    assert all(c["end"] - c["start"] <= 8.0 + 1e-6 for c in cues)
    assert "".join(c["text"] for c in cues) == text


def test_srt_split_keeps_words_at_their_real_time():
    """切完之后每条的时间必须还是那几个字自己的时间，不能靠比例摊。"""
    text = "前半段" + "后半段"
    words = ([{"start": 0.0, "end": 1.0, "word": c} for c in "前半段"]
             + [{"start": 100.0, "end": 101.0, "word": c} for c in "后半段"])
    rows = [{"start": 0.0, "end": 101.0, "text": text, "words": words}]
    cues = render.resplit_rows(rows, max_dur=8.0, max_chars=16)
    assert len(cues) == 2
    assert cues[0]["end"] <= 2.0
    assert cues[1]["start"] >= 99.0


def test_srt_defaults_actually_resplit():
    """锁**默认值**，不只锁显式传参。

    2026-08-07 变异测试发现的空洞：所有字幕测试都显式传 max_dur/max_chars，
    于是把默认值改成 999/9999 时整个测试套件全绿——而那正是坑 12 记录的
    事故本身（最长一条字幕 296 秒 / 128 字，没法逐句回听核对）。
    默认值是真实用户唯一会用到的那组值，必须单独锁。
    """
    text = "这是一段很长的话需要按词的时间切成若干条字幕方便逐句核对原音再多加一些字"
    rows = [{"start": 0.0, "end": 120.0, "text": text,
             "words": even_words(0.0, 120.0, text)}]
    cues = render.resplit_rows(rows)          # ← 不传参数
    assert len(cues) > 1, "默认参数下 120 秒的整段必须被切开"
    assert all(c["end"] - c["start"] <= 10.0 for c in cues), \
        "默认 max_dur 应在 10 秒量级；放大到几百秒等于关掉重切"
    assert "".join(c["text"] for c in cues) == text


def test_srt_cue_always_has_positive_duration():
    """零时长/负时长的字幕条是非法 SRT，播放器会跳过——内容等于丢了。

    防重叠那段把 start 往后推时，可能推过原来的 end。
    """
    rows = [{"start": 0.0, "end": 10.0, "text": "前面这条很长"},
            {"start": 1.0, "end": 1.05, "text": "被夹在中间的极短条"},
            {"start": 1.02, "end": 1.03, "text": "再来一条更短的"}]
    cues = render.resplit_rows(rows)
    for c in cues:
        assert c["end"] > c["start"], f"字幕条时长非正：{c}"


def test_srt_drops_nothing():
    rows = [{"start": 0.0, "end": 5.0, "text": "甲乙丙"},
            {"start": 5.0, "end": 9.0, "text": "丁戊己"}]
    cues = render.resplit_rows(rows)
    assert "".join(c["text"] for c in cues) == "甲乙丙丁戊己"


# ---------------------------------------------------------------- 标点

def test_clause_lead_not_inserted_after_copula():
    """实测 15 处 `这是，第一个环节`——连接词规则没看前一个字。"""
    t = "今天的话课程有三个环节的内容这是第一个环节第二个环节我们会邀请到一位老师"
    out = render.insert_clause_breaks(t)
    assert "是，第" not in out
    assert "了，第" not in out


def test_clause_lead_still_breaks_long_runs():
    t = "我们把这件事情从头到尾再仔细讲一遍接下来要说的是另外一个完全不同的问题"
    out = render.insert_clause_breaks(t)
    assert "遍，接下来" in out, "长串还是要断开，不能因为加了护栏就一刀不切"


def test_word_gaps_become_punctuation():
    """段内 90 字裸奔的根因：停顿标点只作用在段与段之间，段内一个字都不加。"""
    words = ([{"start": 0.0, "end": 1.0, "word": c} for c in "前面这半句"]
             + [{"start": 2.5, "end": 3.5, "word": c} for c in "后面那半句"])
    row = {"start": 0.0, "end": 3.5, "text": "前面这半句后面那半句", "words": words}
    out = render.punctuate_row(row)
    assert out == "前面这半句。后面那半句", "1.5 秒的停顿应该落成句号，且落在两半句之间"


def test_gap_thresholds_adapt_to_a_slow_speaker():
    """慢速歌唱这类慢速音频，词间 0.5 秒是常态，固定 0.9 秒会把句号插进词中间。"""
    slow = [{"start": i * 1.0, "end": i * 1.0 + 0.5, "word": "字"} for i in range(60)]
    cg, pg = render.gap_thresholds([{"words": slow}])
    assert cg >= 0.45, "慢速讲者的逗号阈值必须跟着抬上去"
    assert pg > cg


def test_gap_thresholds_adapt_to_a_fast_speaker():
    fast = [{"start": i * 0.22, "end": i * 0.22 + 0.2, "word": "字"} for i in range(60)]
    cg, pg = render.gap_thresholds([{"words": fast}])
    assert cg <= 0.35, "语速快的讲者，阈值要压下来才断得出逗号"


def test_gap_thresholds_fall_back_when_too_few_words():
    assert render.gap_thresholds([]) == (0.35, 0.90)


def test_punctuate_row_without_words_is_identity():
    row = {"start": 0.0, "end": 3.0, "text": "没有词级时间戳"}
    assert render.punctuate_row(row) == "没有词级时间戳"


# ---------------------------------------------------------------- 保真护栏

def test_speaker_repetition_is_never_collapsed():
    """`业务交流业务交流` 在 pass1 原始输出里就是这样——79/89 处重复源自单段内部，
    是讲者自己的重复，不是合并产生的。自动折叠等于 docs/lessons.md 坑 4 那类静悄悄删真内容。"""
    t = "由我先给大家进行业务交流业务交流就是财务管理"
    out = render.insert_clause_breaks(t)
    assert out.count("业务交流") == 2


def test_merge_keeps_content_unique_to_each_pass():
    """时间重叠的两段不能整段丢，同一时间窗里另一路常有对方漏掉的内容。"""
    rows = [{"start": 0.0, "end": 10.0, "text": "根据大型活动场所财务管理办法第九条规定"},
            {"start": 0.5, "end": 10.5, "text": "根据大型活动场所财务管理办法第九条规定应当指定三人管理"}]
    out = merge.merge_rows(rows)
    joined = "".join(r["text"] for r in out)
    assert "应当指定三人管理" in joined
    assert joined.count("第九条规定") == 1


def test_merge_keeps_repeated_citations_when_times_do_not_overlap():
    """法规名反复引用是正常的，不能按字块裁成`根据大九条`。"""
    name = "根据大型活动场所财务管理办法第九条"
    rows = [{"start": 0.0, "end": 5.0, "text": name + "我们要做资产登记"},
            {"start": 20.0, "end": 25.0, "text": name + "还要做年度报告"}]
    out = merge.merge_rows(rows)
    joined = "".join(r["text"] for r in out)
    assert joined.count(name) == 2


def test_starved_row_is_backfilled_without_duplicating_itself():
    """饥饿段会被切成多个补转区段，而它本身跨越所有区段、不会被整行替换掉。

    所以补回来的内容必然和原行时间重叠——要靠 merge_rows 去重，
    否则那 128 字会连同补转结果一起重复出现在成品里。
    """
    kept = "甲乙丙丁戊己庚辛壬癸"
    full = kept + "子丑寅卯辰巳午未申酉戌亥这些都是原本被吞掉没转出来的内容"
    passes = [{"duration": 300.0,
               "segments": [seg(0.0, 300.0, kept, logprob=-0.28)]}]
    patch = [{"span": [0.0, 90.0], "label": "段内饥饿",
              "attempts": [{"mode": "novad",
                            "rows": [{"start": 0.0, "end": 60.0, "text": full}]}]}]
    rows = merge.combine(passes, patch, {}, [], 300.0)
    joined = "".join(r["text"] for r in rows)
    assert "这些都是原本被吞掉没转出来的内容" in joined, "补转内容必须进得来"
    assert joined.count(kept) == 1, f"原行不该和补转结果重复出现：{joined}"


# ---------------------------------------------------------------- 术语表

def test_clause_break_survives_after_final_particle():
    """的/了 是句末助词，它们后面恰恰该断句——把它们当粘连字会挡掉正确的逗号。"""
    t = "这个会计科目的使用范围已经明显缩小了所以说新旧制度衔接的时候一定要重新分类处理"
    out = render.insert_clause_breaks(t)
    assert "了，所以说" in out


def test_terms_report_counts_hits():
    """README 教人手工统计频次再加条目，这件事该由代码来做。"""
    terms = {"fixes": [["才税管理", "财税管理"], ["从来不会出现的词", "X"]]}
    text = "才税管理的才税管理讲座"
    hits = render.terms_hits(text, terms)
    assert hits["才税管理"] == 2
    assert hits["从来不会出现的词"] == 0
