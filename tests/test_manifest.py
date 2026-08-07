import json

from whisperaudit.bench import manifest as M


def test_read_manifest_skips_blank_lines(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"audio":"a.wav","text":"甲乙"}\n\n{"audio":"b.wav","text":"丙丁"}\n',
                 encoding="utf-8")
    items = M.read_manifest(str(p))
    assert [i["text"] for i in items] == ["甲乙", "丙丁"]


def test_eval_manifest_aggregates_over_items():
    items = [{"audio": "a.wav", "text": "旧账导进去"},
             {"audio": "b.wav", "text": "会计凭证"}]
    fake = {"a.wav": "旧帐导进去", "b.wav": "会计凭证"}
    rep = M.eval_manifest(items, lambda p: fake[p])
    assert rep["n_items"] == 2
    assert rep["sub"] == 1
    assert rep["homo"] == 1
    assert rep["cer"] > 0


def test_eval_manifest_on_perfect_hypotheses_is_zero():
    items = [{"audio": "a.wav", "text": "甲乙丙"}]
    rep = M.eval_manifest(items, lambda p: "甲乙丙")
    assert rep["cer"] == 0.0


def test_eval_manifest_weights_by_characters_not_by_item():
    """按字加权，不是逐条平均。

    一条 100 字全对、一条 2 字全错，逐条平均会得到 50% 的荒谬结果；
    按字加权应当是 2/102 ≈ 2%。
    """
    items = [{"audio": "long.wav", "text": "字" * 100},
             {"audio": "short.wav", "text": "甲乙"}]
    fake = {"long.wav": "字" * 100, "short.wav": "丙丁"}
    rep = M.eval_manifest(items, lambda p: fake[p])
    assert rep["cer"] == 2 / 102
    assert rep["cer"] < 0.05, "若接近 0.5 说明退化成了逐条平均"


def test_eval_manifest_tolerates_empty_reference_rows():
    """n_ref=0 的条目其 cer 恒为 0，逐条平均会被它拉低；累加则不受影响。"""
    items = [{"audio": "a.wav", "text": ""},
             {"audio": "b.wav", "text": "甲乙丙丁"}]
    fake = {"a.wav": "幻觉文字", "b.wav": "甲乙丙戊"}
    rep = M.eval_manifest(items, lambda p: fake[p])
    assert rep["ins"] == 4      # 空参考对应的 4 个字算插入
    assert rep["sub"] == 1
    assert rep["cer"] == (1 + 4) / 4


def test_eval_manifest_keeps_per_item_hypotheses():
    """转录烧 GPU 小时，评分口径调整不该逼人重烧——逐条 hyp 必须带回。"""
    items = [{"audio": "a.wav", "text": "甲乙"}]
    rep = M.eval_manifest(items, lambda p: "甲丙")
    assert rep["hyps"] == [{"audio": "a.wav", "text": "甲乙", "hyp": "甲丙"}]
    lean = M.eval_manifest(items, lambda p: "甲丙", keep_hyps=False)
    assert "hyps" not in lean


def test_eval_manifest_cleans_reference_glosses_by_default():
    """公开基准的参考多取自书面文本，含说话人不会读的外文注释。"""
    items = [{"audio": "a.wav", "text": "总统埃尔多安（Recep Erdoğan）发表声明"}]
    rep = M.eval_manifest(items, lambda p: "总统埃尔多安发表声明")
    assert rep["cer"] == 0.0, "注释未剔除会凭空产生删除错"
    assert rep["ref_cleaned"] > 0

    raw = M.eval_manifest(items, lambda p: "总统埃尔多安发表声明", clean_ref=False)
    assert raw["dele"] > 0 and raw["ref_cleaned"] == 0
