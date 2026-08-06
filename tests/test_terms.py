import json
from pathlib import Path

import pytest

from audio_transcribe import terms as T

TERMS = {"terms": ["财税管理", "大型活动场所", "非营利组织", "记账凭证"]}

REAL_TERMS_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples" / "terms" / "finance-lecture.json"
)


def test_exact_homophone_is_corrected():
    out, hits = T.pinyin_fix("才税管理的要点", TERMS)
    assert out == "财税管理的要点"
    assert hits[0]["from"] == "才税管理" and hits[0]["to"] == "财税管理"


def test_near_homophone_is_corrected_when_loose():
    """厂所(chang suo) vs 场所(chang suo)——同音；中大 vs 大型靠近音。"""
    out, _ = T.pinyin_fix("大型活动厂所登记", TERMS)
    assert out == "大型活动场所登记"


def test_correct_text_is_left_alone():
    out, hits = T.pinyin_fix("非营利组织的记账凭证", TERMS)
    assert out == "非营利组织的记账凭证"
    assert hits == []


def test_different_pinyin_is_never_touched():
    """锁死：拼音不同的词一律不许改动。这是防止静悄悄改内容的护栏。"""
    src = "会计科目和固定资产的处理"
    out, hits = T.pinyin_fix(src, TERMS)
    assert out == src
    assert hits == []


def test_longer_term_wins_over_shorter():
    terms = {"terms": ["场所", "大型活动场所"]}
    out, _ = T.pinyin_fix("大型活动厂所", terms)
    assert out == "大型活动场所"


def test_hits_record_position_for_accounting():
    """所有改动必须可追溯，写进质检报告。"""
    _, hits = T.pinyin_fix("前面才税管理后面", TERMS)
    assert hits[0]["pos"] == 2


def test_empty_terms_is_noop():
    assert T.pinyin_fix("任何文本", {}) == ("任何文本", [])


def test_real_termlist_does_not_touch_unrelated_text():
    """过度触发验证（brief 未要求，自行加）：

    真实术语表的 terms（当前 44 条） 覆盖财税/大型场所行政词汇，与日常口语在拼音上
    没有交集。拿一段不含任何术语的普通中文过一遍，必须一个字都不改——
    这是拼音模糊匹配最大的风险（误伤正常文本）的护栏。
    """
    real_terms = json.loads(REAL_TERMS_PATH.read_text(encoding="utf-8"))
    assert real_terms["terms"], "术语表不能为空，否则这条护栏形同虚设"

    daily_text = (
        "今天天气很好，我和朋友一起去公园散步，路上买了几个包子当早饭。"
        "下午我们去图书馆看书，晚上回家煮了一锅汤，加了萝卜和排骨，"
        "味道很鲜美。周末打算去爬山，顺便拍些照片留作纪念，"
        "路上还遇到了几只小狗。"
    )

    out, hits = T.pinyin_fix(daily_text, real_terms)
    assert out == daily_text
    assert hits == []


def test_short_term_must_not_corrupt_correct_text_across_word_boundary():
    """短术语跨词边界误匹配，会把本来正确的文本改坏。

    2026-08-05 在 41,174 字真实转录上实测撞到：

        原文  …大型活动场所的税务登记义务…      本来就是对的
        误改  …大型活动场所得税务登记义务…      「所的税」被当成「所得税」的同音错

    「所得税」这条术语单独存在时，扫描器会在「场所|的税务」的接缝上凑出
    「所的税」并替换掉。真实语料里没出事，纯粹因为术语表里恰好还有更长的
    「大型活动场所」，长词优先把这几个字先吃掉了——**是巧合，不是设计**。

    这条测试锁住这个巧合：真实术语表必须始终能保住这句话不被改动。
    """
    real_terms = json.loads(REAL_TERMS_PATH.read_text(encoding="utf-8"))
    src = "这一规定将大型活动场所的税务登记义务也纳入管理"
    for loose in (False, True):
        out, hits = T.pinyin_fix(src, real_terms, loose=loose)
        assert out == src, f"loose={loose} 改坏了正确文本：{out}（{hits}）"


def test_boundary_hazard_is_reproducible_without_the_longer_term():
    """把上面那个巧合摘掉，证明危险是真实存在的，不是杞人忧天。

    这条测试**故意断言坏行为**，作用是：一旦将来给 pinyin_fix 加了真正的
    词边界保护，它会失败，提醒把上面那条护栏和文档一起更新。
    """
    src = "这一规定将大型活动场所的税务登记义务也纳入管理"
    out, hits = T.pinyin_fix(src, {"terms": ["所得税"]}, loose=False)
    assert out != src, "若这条开始通过，说明已有词边界保护，请更新文档与护栏"
    assert hits[0]["from"] == "所的税"
