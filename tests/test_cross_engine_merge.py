"""跨引擎合并：把 combine() 该有的行为先锁住。

背景：当前两路都是 whisper（只是 chunk 长度不同），计划换一路成中文专用的
第二引擎（跨引擎互补），但那个引擎暂时因环境问题装不上。这些测试锁住的是
**跨引擎场景下 combine() 现在实际的行为**——等引擎能用时，直接拿真实输出
跑这些用例，看是否还符合预期，不必等到那时才第一次思考这些边界情况。

无 GPU、无音频、纯文本。

    python3 -m pytest tests/test_cross_engine_merge.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisper_audit import merge as M


def wl(start, end, text):
    """把 text 均匀铺在 [start,end] 上的词级时间戳，仅测试用。"""
    n = len(text)
    step = (end - start) / n
    return [{"start": round(start + i * step, 3),
              "end": round(start + (i + 1) * step, 3), "word": c}
             for i, c in enumerate(text)]


# ---------------------------------------------------------------- brief 里指定的两条

def test_cross_engine_merge_keeps_content_unique_to_each():
    """跨引擎分段差异更大，不能因为「看起来重复」就丢掉另一路独有的内容。"""
    a = {"duration": 60.0, "segments": [
        {"start": 0.0, "end": 30.0, "text": "现金箱应当指定三人管理"}]}
    b = {"duration": 60.0, "segments": [
        {"start": 0.0, "end": 12.0, "text": "现金箱应当指定三人管理"},
        {"start": 12.0, "end": 30.0, "text": "开启时三人同时在场当场清点登记签字"}]}
    rows = M.combine([a, b], [], {}, [], 60.0)
    joined = "".join(r["text"] for r in rows)
    assert "开启时三人同时在场" in joined
    assert joined.count("现金箱应当指定三人管理") == 1


def test_engine_without_words_does_not_break_downstream():
    """第二引擎很可能给不出词级时间戳——下游必须能降级而不是崩。"""
    a = {"duration": 30.0, "segments": [
        {"start": 0.0, "end": 10.0, "text": "没有词级时间戳的引擎"}]}
    rows = M.combine([a], [], {}, [], 30.0)
    assert rows and rows[0].get("words", []) == []


# ---------------------------------------------------------------- 签名细节（brief 写不到的坑）

def test_pinyin_correction_stays_off_by_default():
    """pinyin 默认已改为 False——不传该参数时，不能把「节余」纠成同音的「结余」。

    这两个词读音全同、意思不同，是拼音纠错默认开启时真实误伤过的例子。
    combine() 不显式传 pinyin=True 就必须原样放行。
    """
    a = {"duration": 30.0, "segments": [
        {"start": 0.0, "end": 5.0, "text": "本月节余较多"}]}
    terms = {"terms": ["结余"]}   # 术语表里只有「结余」，同音但字不同
    rows = M.combine([a], [], terms, [], 30.0)
    assert rows[0]["text"] == "本月节余较多"


def test_return_hits_toggles_return_shape():
    """不传 return_hits 时返回 list；传了才返回 (rows, hits) 二元组。

    这两种返回形状调用方一旦弄混，会把 hits 列表当 rows 用，或者反过来
    把某一行拆散成两个字段读——所以必须显式钉住类型。
    """
    a = {"duration": 30.0, "segments": [
        {"start": 0.0, "end": 5.0, "text": "返回形状测试"}]}
    rows = M.combine([a], [], {}, [], 30.0)
    assert isinstance(rows, list) and isinstance(rows[0], dict)

    out = M.combine([a], [], {}, [], 30.0, return_hits=True)
    assert isinstance(out, tuple) and len(out) == 2
    rows2, hits = out
    assert isinstance(rows2, list) and isinstance(hits, list)
    assert hits == []   # pinyin=False 时不会产生任何改动记录


def test_words_none_is_treated_like_missing():
    """某些引擎适配层可能显式写 "words": None 而不是省略或给空列表——两者要等价。"""
    a = {"duration": 30.0, "segments": [
        {"start": 0.0, "end": 5.0, "text": "词表显式为None", "words": None}]}
    rows = M.combine([a], [], {}, [], 30.0)
    assert rows[0].get("words") == []


# ---------------------------------------------------------------- 我自己加的对抗性测试

def test_mixed_words_presence_across_engines_does_not_crash():
    """一路有 words、另一路没有，混在一起：不同 30 秒窗口各自选中不同的一路，
    有词表的窗口保留词表，没词表的窗口不能报错、也不能凭空造出词表。"""
    a_text0 = "甲引擎第一窗口带词表内容较多用于获胜"
    a_text1 = "甲引擎第二窗口较短"
    a = {"duration": 60.0, "segments": [
        {"start": 0.0, "end": 30.0, "text": a_text0, "words": wl(0.0, 30.0, a_text0)},
        {"start": 30.0, "end": 60.0, "text": a_text1, "words": wl(30.0, 60.0, a_text1)},
    ]}
    b = {"duration": 60.0, "segments": [
        {"start": 0.0, "end": 30.0, "text": "乙引擎第一窗口短"},          # 没有 words 字段，且更短，输掉第一窗口
        {"start": 30.0, "end": 60.0, "text": "乙引擎第二窗口带更多内容因此在这个窗口获胜"},  # 更长，赢下第二窗口
    ]}
    rows = M.combine([a, b], [], {}, [], 60.0)
    assert len(rows) == 2
    first, second = sorted(rows, key=lambda r: r["start"])
    assert first["text"] == a_text0
    assert first["words"], "甲引擎赢的窗口必须带着它自己的词表"
    assert second["text"] == "乙引擎第二窗口带更多内容因此在这个窗口获胜"
    assert second.get("words", []) == [], "乙引擎赢的窗口没有词表，不能凭空造一个出来"


def test_words_dropped_when_boundary_text_gets_trimmed():
    """merge_rows 裁字块去重时，被裁过的行必须扔掉词表而不是留着错位的时间戳——
    详见 merge.py 里 `row.pop("words", None)` 那条注释：错位的词级时间戳
    比没有更误导。这里直接钉住：文本一旦被裁，"words" 键必须整个消失。"""
    prev_text = "根据大型活动场所财务管理办法第九条规定"
    cur_text = "根据大型活动场所财务管理办法第九条规定应当指定三人管理"
    rows = M.merge_rows([
        {"start": 0.0, "end": 10.0, "text": prev_text, "words": wl(0.0, 10.0, prev_text)},
        {"start": 0.5, "end": 10.5, "text": cur_text, "words": wl(0.5, 10.5, cur_text)},
    ])
    assert len(rows) == 2
    trimmed = rows[1]
    assert trimmed["text"] == "应当指定三人管理"
    assert "words" not in trimmed, "文本被裁过之后，错位的词表必须整个消失，不能留一个空列表糊弄下游"


def test_bucket_winner_take_all_can_silently_drop_non_overlapping_content():
    """已知缺口，先用测试钉住现状，不代表这是我们想要的行为。

    combine() 按 30 秒窗口整窗二选一（谁的字数多选谁），不是逐句/逐字合并。
    当跨引擎切分粒度差异很大时，这个「整窗归属一路」的规则会出问题：
    输的那一路即使在窗口里某个子区间有对方完全没有、且和赢家文本毫无重叠的
    真实内容，也会被**整窗**丢弃——这不是「误判成重复被删」，是压根没进入
    候选就被扔掉了，比误判去重更彻底。

    这里 pass A 一句话覆盖整个 30 秒窗口、字数占优，赢下整窗；pass B 只在
    窗口两端各有一小段，末尾那句「留存原始凭证」是 A 完全没提到的独有内容，
    随着 B 整体输掉窗口一起消失。跨引擎合并如果要做到「不遗漏」，这条大概率
    要重新设计（比如窗口内做逐段合并而不是整窗二选一），但那是另一个任务；
    这里先把当前行为写成测试，回归时一旦意外改变会立刻被看见。
    """
    a = {"duration": 60.0, "segments": [
        {"start": 0.0, "end": 30.0,
         "text": "现金箱应当指定三人共同管理并按月公示账目防止出现挪用问题"}]}
    b = {"duration": 60.0, "segments": [
        {"start": 0.0, "end": 5.0, "text": "现金箱应当指定三人"},
        {"start": 25.0, "end": 30.0, "text": "还要留存原始凭证备查"}]}
    rows = M.combine([a, b], [], {}, [], 60.0)
    joined = "".join(r["text"] for r in rows)
    assert "留存原始凭证" not in joined, (
        "现状如此：B 输掉整窗，它独有的内容也一并丢失。"
        "这行断言如果哪天变成失败，说明 combine() 已经改成逐段合并了，"
        "那是好事，请把这条测试整个删掉或者改成正面断言。")


def test_fine_grained_engine_can_fully_win_a_window_with_all_its_segments_intact():
    """反过来：细粒度那一路如果总字数占优，会整窗胜出，且它自己内部的多个
    小段要原样都在，不能因为「赢了」就只挑其中一段。"""
    a = {"duration": 60.0, "segments": [
        {"start": 0.0, "end": 30.0, "text": "现金箱应当指定三人管理"}]}
    b_segs = [
        ("现金箱应当指定三人管理", 0.0, 5.0),
        ("开启时三人同时在场", 5.0, 10.0),
        ("当场清点登记签字", 10.0, 15.0),
        ("并留存原始凭证", 15.0, 20.0),
        ("按月公示账目", 20.0, 25.0),
        ("防止出现挪用问题", 25.0, 30.0),
    ]
    b = {"duration": 60.0, "segments": [
        {"start": s, "end": e, "text": t} for t, s, e in b_segs]}
    rows = M.combine([a, b], [], {}, [], 60.0)
    assert len(rows) == len(b_segs)
    joined = "".join(r["text"] for r in sorted(rows, key=lambda r: r["start"]))
    assert joined == "".join(t for t, _, _ in b_segs)


def test_tie_break_between_engines_favors_the_later_pass():
    """两路对同一窗口字数刚好相等时（真实字数一样、内容却分歧），选哪一路完全
    由传入顺序决定：后传入的赢。这是 `max((n, i, rows) for i, ...)` 元组比较
    的副作用，不是有意为之的"更信任第二引擎"，但既然是确定性行为就该钉住——
    以后调 PROFILES 里 engines 列表顺序时要知道这一条在起作用。"""
    a = {"duration": 30.0, "segments": [
        {"start": 0.0, "end": 30.0, "text": "现金箱应当指定三人管理"}]}
    b = {"duration": 30.0, "segments": [
        {"start": 0.0, "end": 30.0, "text": "损款箱因当指定山人管理"}]}
    assert len(a["segments"][0]["text"]) == len(b["segments"][0]["text"]), \
        "这条测试的前提就是字数相等，制造真正的平局"

    rows_ab = M.combine([a, b], [], {}, [], 30.0)
    assert rows_ab[0]["text"] == b["segments"][0]["text"], "[a, b] 顺序下后者 b 赢"

    rows_ba = M.combine([b, a], [], {}, [], 30.0)
    assert rows_ba[0]["text"] == a["segments"][0]["text"], "[b, a] 顺序下后者 a 赢"


def test_completely_different_transcriptions_for_same_span():
    """两路对同一时间段给出完全不同的文本（不是一方漏字，是识别结果分歧）。

    这不是"重复因此去重"的场景——是两个引擎对同一段音频给出了完全不同的
    识别结果。当前 combine() 并不真正"裁决分歧"：它只是把整窗当成一次
    字数竞赛，字多的那一路整段胜出，字少的那一路（哪怕它才是对的）完全消失，
    不会被记录、不会被标记为"有分歧待复核"。

    这是现状，不一定是最优——更合理的做法也许是把这类高度分歧的窗口标记出来
    交给人工复核，而不是静默二选一。这里先钉住现状，避免以后改动 combine()
    时在没人注意的情况下把这条隐藏的假设改掉。
    """
    a = {"duration": 30.0, "segments": [
        {"start": 0.0, "end": 30.0, "text": "现金箱应当指定三人管理"}]}
    b = {"duration": 30.0, "segments": [
        {"start": 0.0, "end": 30.0, "text": "存钱罐由三个和尚共同看管每月对账一次"}]}
    rows = M.combine([a, b], [], {}, [], 30.0)
    assert len(rows) == 1
    assert rows[0]["text"] == b["segments"][0]["text"], (
        "现状：字数更多的一路整段胜出，另一路的说法不会以任何形式留痕")
