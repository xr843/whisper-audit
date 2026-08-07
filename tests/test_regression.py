"""CI 回归门禁：固定文本对的 CER 指标必须稳定。

CI 跑不了 ASR（无 GPU、无音频），所以门禁只盯**纯文本阶段**：
评测工具本身的输出稳定性，以及拼音纠错在固定输入上的表现。
这是诚实的边界——它不假装覆盖了识别环节。

夹具里每一条对应一类真实错误模式，内容为构造，不含任何敏感信息。
"""
import json
import pathlib

import pytest

from whisper_audit import evaluate as E
from whisper_audit import terms as T

FIX = pathlib.Path(__file__).parent / "fixtures" / "regression.jsonl"


def load():
    return {json.loads(l)["id"]: json.loads(l)
            for l in FIX.read_text(encoding="utf-8").splitlines() if l.strip()}


@pytest.fixture(scope="module")
def scored():
    return {k: E.score(v["ref"], v["hyp"]) for k, v in load().items()}


def test_clean_pair_scores_zero(scored):
    assert scored["clean"]["cer"] == 0.0
    assert scored["clean"]["sub"] == 0


def test_pure_deletion_is_counted_as_deletion(scored):
    """删除率是「不遗漏」的直接度量，不能被算成替换。"""
    s = scored["deletion"]
    assert (s["dele"], s["sub"], s["ins"]) == (9, 0, 0)


def test_pure_insertion_is_counted_as_insertion(scored):
    """插入多于参考长度时 cer 会大于 1——这是标准 CER 的正常行为，不是 bug。"""
    s = scored["insertion"]
    assert (s["ins"], s["sub"], s["dele"]) == (6, 0, 0)
    assert s["cer"] == 1.5


def test_homophone_substitutions_are_recognised(scored):
    s = scored["homophone"]
    assert s["sub"] == 2 and s["homo"] == 2


def test_near_homophone_is_recognised(scored):
    """厂/场 同音，属 homo；homo 必须同时计入 near。"""
    s = scored["near_accent"]
    assert s["sub"] == 1 and s["homo"] == 1 and s["near"] == 1


def test_latin_substitution_is_never_counted_as_homophone(scored):
    """pinyin_key 对非中文返回空元组，两个空元组相等曾让 APP→ABC 被判 100% 同音，
    静默抬高 homo_pct——而 homo_pct 是决定要不要做同音修正的依据。"""
    s = scored["latin"]
    assert s["sub"] == 2
    assert s["homo"] == 0 and s["near"] == 0


def test_cer_denominator_is_reference_length(scored):
    for k, s in scored.items():
        expect = (s["sub"] + s["dele"] + s["ins"]) / s["n_ref"] if s["n_ref"] else 0.0
        assert s["cer"] == pytest.approx(expect), f"{k} 的 cer 分母不是参考长度"


def test_near_always_includes_homo(scored):
    for k, s in scored.items():
        assert s["near"] >= s["homo"], f"{k}: near 必须包含 homo"


def test_pinyin_fix_improves_cer_on_the_fixture():
    """拼音纠错必须真的降低 CER，而不是「装上了就开着」。"""
    fx = load()["near_accent"]
    terms = {"terms": ["大型活动场所"]}
    before = E.score(fx["ref"], fx["hyp"])["cer"]
    fixed, hits = T.pinyin_fix(fx["hyp"], terms)
    after = E.score(fx["ref"], fixed)["cer"]
    assert hits, "该修的没修"
    assert after < before
    assert after == 0.0


def test_pinyin_fix_does_not_touch_the_clean_pair():
    """对已经正确的文本，拼音纠错必须一个字都不改。"""
    fx = load()["clean"]
    terms = {"terms": ["记账凭证", "原始凭证", "会计凭证"]}
    fixed, hits = T.pinyin_fix(fx["hyp"], terms)
    assert fixed == fx["hyp"]
    assert hits == []
