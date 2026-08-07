import pytest

from whisper_audit import evaluate as E


def test_normalize_strips_punctuation_and_converts_traditional():
    assert E.normalize("上課的時間，差不多到了。") == "上课的时间差不多到了"


def test_normalize_unifies_digits():
    assert E.normalize("第9条") == E.normalize("第九条")


def test_score_counts_substitution_deletion_insertion():
    r = E.score("甲乙丙丁", "甲X丙丁戊")      # 乙→X 替换，末尾多一个戊
    assert r["sub"] == 1
    assert r["ins"] == 1
    assert r["dele"] == 0
    assert r["cer"] == pytest.approx(0.5, abs=1e-9)


def test_score_deletion_is_reported_separately():
    """删除率是「不遗漏」这个卖点的直接度量，必须单列。"""
    r = E.score("甲乙丙丁戊己", "甲乙丙")
    assert r["dele"] == 3
    assert r["sub"] == 0


def test_homophone_substitution_is_measured():
    """旧账→旧帐：账/帐 同音，这类错拼音纠错能修。"""
    r = E.score("旧账导进去", "旧帐导进去")
    assert r["sub"] == 1
    assert r["homo"] == 1


def test_near_homophone_catches_accent_confusion():
    """南方方言口音把 zh 读成 z：账(zhang)→赃(zang) 不同音但近音。"""
    r = E.score("旧账", "旧赃")
    assert r["sub"] == 1
    assert r["homo"] == 0
    assert r["near"] == 1


def test_unrelated_error_is_neither_homophone_nor_near():
    r = E.score("旧账", "旧猫")
    assert r["sub"] == 1
    assert r["homo"] == 0
    assert r["near"] == 0


def test_perfect_match_scores_zero():
    r = E.score("完全一样的文本", "完全一样的文本")
    assert r["cer"] == 0.0
    assert r["homo_pct"] == 0.0


def test_latin_substitution_is_not_counted_as_homophone():
    """APP→ABC 是无关替换；两侧都转不出拼音时 pinyin_key 都返回空元组，
    空元组彼此相等会被误判成同音，必须用守卫挡住（回归用例，Critical 1）。"""
    r = E.score("这是APP的问题", "这是ABC的问题")
    assert r["sub"] == 2
    assert r["homo"] == 0
    assert r["near"] == 0


def test_mixed_chinese_latin_homophone_still_detected():
    """中英混排时，中文部分的同音判定不能被同一条错误里的英文噪声连累。"""
    r = E.score("旧账APP", "旧帐ABC")
    assert r["sub"] == 3
    assert r["homo"] == 1
    assert r["near"] == 1


def test_hanzi_digit_styles_are_equivalent():
    """「二零一九」和「2019」是同一个数的两种合法读写。

    曾经 0 映射为「〇」而 ASR 引擎都写「零」，「二零一九」对「2019」平白
    多一个替换错——系统性偏袒输出阿拉伯数字的引擎（FLEURS 双引擎对比撞出）。
    """
    assert E.score("2019年", "二零一九年")["cer"] == 0.0
    assert E.score("二〇一九", "二零一九")["cer"] == 0.0


def test_strip_foreign_gloss_removes_written_only_annotations():
    """括号内的纯外文注释是书面约定，说话人不会读——参考里留着它，
    任何 ASR 都会凭空吃满删除错。实测 FLEURS 上这一项让 whisper 的
    CER 从 3.77% 虚高到 7.56%（近两倍），且各引擎受损不等，对比失真。"""
    ref = "特朗普与土耳其总统埃尔多安（Recep Tayyip Erdoğan）通话后发表了声明"
    assert E.strip_foreign_gloss(ref) == "特朗普与土耳其总统埃尔多安通话后发表了声明"


def test_strip_foreign_gloss_keeps_chinese_parentheses():
    """含中文的括号不动——那是说话人可能真读出来的内容。"""
    for keep in ("甲方（以下简称甲方）应当", "会议（第三次）已召开"):
        assert E.strip_foreign_gloss(keep) == keep


def test_strip_foreign_gloss_is_a_noop_without_glosses():
    plain = "这段文本没有任何括号注释"
    assert E.strip_foreign_gloss(plain) == plain
