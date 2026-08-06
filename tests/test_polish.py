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
