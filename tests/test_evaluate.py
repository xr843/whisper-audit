import pytest

from audio_transcribe import evaluate as E


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
    """方言口音把 zh 读成 z：账(zhang)→赃(zang) 不同音但近音。"""
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
