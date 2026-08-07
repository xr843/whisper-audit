"""run_polish 的护栏测试。

这个函数此前零覆盖，于是它无条件跑拼音纠错、丢掉记账、连 dry-run 都改正文——
把项目最看重的「不静悄悄改内容」和「全量记账」两条同时破掉了。
"""
from types import SimpleNamespace

from whisperaudit.cli import run_polish

TERMS = {"terms": ["结余"], "fixes": []}


def mkargs(**kw):
    base = dict(polish=True, polish_dry_run=False, pinyin_fix=False,
                loose_pinyin=False, llm_base_url="http://unused",
                llm_model="unused")
    base.update(kw)
    return SimpleNamespace(**base)


def rows_of(text):
    return [{"start": 0.0, "end": 3.0, "text": text}]


def test_dry_run_changes_nothing():
    """dry-run 承诺「只打印将要发送什么」，就必须一个字都不改。

    曾经它会把「节余」改成「结余」——正是四处文档拿来当
    「拼音纠错必须默认关」招牌反例的那个词——而且记账为 0。
    """
    rows = rows_of("本年度节余资金五万元")
    rep, hits = run_polish(rows, TERMS, mkargs(polish=False, polish_dry_run=True))
    assert rows[0]["text"] == "本年度节余资金五万元"
    assert hits == []


def test_dry_run_still_changes_nothing_even_if_pinyin_fix_is_on():
    rows = rows_of("本年度节余资金五万元")
    run_polish(rows, TERMS, mkargs(polish=False, polish_dry_run=True, pinyin_fix=True))
    assert rows[0]["text"] == "本年度节余资金五万元"


def test_fallback_pinyin_fix_requires_explicit_opt_in(monkeypatch):
    """没开 --pinyin-fix 时，兜底不该跑——没有可覆盖的对象。"""
    monkeypatch.setenv("WHISPERAUDIT_LLM_KEY", "x")
    monkeypatch.setattr("whisperaudit.polish.polish",
                        lambda text, **kw: (text, dict(chunks=1, accepted=0, rejected=0,
                                                       length_rejected=0, failed=0)))
    rows = rows_of("本年度节余资金五万元")
    _, hits = run_polish(rows, TERMS, mkargs(pinyin_fix=False))
    assert rows[0]["text"] == "本年度节余资金五万元"
    assert hits == []


def test_fallback_pinyin_fix_is_accounted_when_enabled(monkeypatch):
    """开了就跑，但每一处改动都必须带时间戳进账本。"""
    monkeypatch.setenv("WHISPERAUDIT_LLM_KEY", "x")
    monkeypatch.setattr("whisperaudit.polish.polish",
                        lambda text, **kw: (text, dict(chunks=1, accepted=0, rejected=0,
                                                       length_rejected=0, failed=0)))
    rows = rows_of("本年度节余资金五万元")
    _, hits = run_polish(rows, TERMS, mkargs(pinyin_fix=True))
    assert rows[0]["text"] == "本年度结余资金五万元"
    assert len(hits) == 1
    assert hits[0]["to"] == "结余" and "t" in hits[0]


def test_structure_rejection_when_line_count_changes(monkeypatch):
    """LLM 把两行并成一行，长度可能恰好不变，constrain 看不出来——行数能。"""
    monkeypatch.setenv("WHISPERAUDIT_LLM_KEY", "x")
    monkeypatch.setattr("whisperaudit.polish.polish",
                        lambda text, **kw: (text.replace("\n", ""),
                                            dict(chunks=1, accepted=0, rejected=0,
                                                 length_rejected=0, failed=0)))
    rows = [{"start": 0.0, "end": 3.0, "text": "第一行"},
            {"start": 3.0, "end": 6.0, "text": "第二行"}]
    rep, _ = run_polish(rows, {"terms": [], "fixes": []}, mkargs())
    assert rep.get("structure_rejected") is True
    assert [r["text"] for r in rows] == ["第一行", "第二行"], "结构被破坏时应整体作废"
