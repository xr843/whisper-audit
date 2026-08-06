from audio_transcribe import polish as P


def test_homophone_edit_is_accepted():
    """帐/账 同音（均 zhang）——brief 原示例用的「港」实测并非「账」的同音字
    （gang vs zhang，声母不同，loose 模式也不等），已换成真同音字对，
    详见 task-10-report.md。"""
    out, rep = P.constrain("旧帐倒进去", "旧账倒进去")
    assert out == "旧账倒进去"
    assert rep["rejected"] == 0


def test_non_homophone_edit_is_reverted():
    """锁死：LLM 改了一个不同音的字，必须还原并记账。"""
    out, rep = P.constrain("旧港倒进去", "旧猫倒进去")
    assert out == "旧港倒进去"
    assert rep["rejected"] == 1


def test_length_change_rejects_whole_chunk():
    """长度变了说明 LLM 增删了内容——整块拒绝，绝不部分接受。"""
    out, rep = P.constrain("旧港倒进去", "旧账导进去了")
    assert out == "旧港倒进去"
    assert rep["reason"] == "length"


def test_deletion_is_rejected():
    out, rep = P.constrain("甲乙丙丁", "甲乙丁")
    assert out == "甲乙丙丁"
    assert rep["reason"] == "length"


def test_mixed_edits_keep_only_the_homophone_ones():
    out, rep = P.constrain("旧帐和固定资产", "旧账和固定资全")
    assert out == "旧账和固定资产"
    assert rep["rejected"] == 1


def test_punctuation_change_is_reverted():
    out, _ = P.constrain("甲乙，丙丁", "甲乙。丙丁")
    assert out == "甲乙，丙丁"


def test_identical_text_is_noop():
    out, rep = P.constrain("完全一样", "完全一样")
    assert out == "完全一样" and rep["rejected"] == 0


def test_non_chinese_edit_is_rejected():
    """pinyin_key 对非中文字符返回空元组，两个空元组相等——不加守卫会把
    字母/数字的任意替换误判成同音放行。这个坑刚在 evaluate.py 里修过一次。"""
    out, rep = P.constrain("会议室A", "会议室B")
    assert out == "会议室A"
    assert rep["rejected"] == 1


def test_chunks_have_a_hard_cap_without_punctuation():
    """ASR 稿常常整段无句末标点——只按标点切会把两万字灌成一整块。"""
    text = "字" * 20000
    parts = P._chunks(text, 1200)
    assert all(len(p) <= 1800 for p in parts), \
        f"最长块 {max(len(p) for p in parts)} 字，硬上限失效"
    assert "".join(parts) == text


def test_chunks_still_prefer_sentence_boundaries():
    text = ("这是一句话。" * 250)          # 1500 字，句号充足
    parts = P._chunks(text, 1200)
    assert all(p.endswith("。") for p in parts[:-1] if p)
    assert "".join(parts) == text


def test_polish_whitespace_padding_does_not_trigger_length_rejection(monkeypatch):
    """行首/行尾空白不能让长度校验必然失败——那会静默压低接受率。"""
    monkeypatch.setattr(P, "_call", lambda *a, **k: "旧账倒进去")
    out, rep = P.polish("  旧帐倒进去 ", base_url="x", model="y", api_key="k")
    assert out == "  旧账倒进去 "
    assert rep["length_rejected"] == 0
    assert rep["accepted"] == 1
