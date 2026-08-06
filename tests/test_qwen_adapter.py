"""Qwen 适配器测试。核心夹具取自真机探测的原始截取，不是照文档推测的形状——
2026-08-06 用 5 分钟真实中文语音探测 Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B
（RTX 4060 8GB）。

实测格式：
  raw = {"text": "大中，请起立。…"（模型自带标点）,
         "items": [{"text": "大", "start": 53.2, "end": 53.28}, …]}  # 秒，非毫秒
  items 是 text 去掉标点后的有序子序列（ForcedAligner 分词器丢弃标点），不能直接
  zip 回 text——5 分钟样本里 text 740 字、items 只有 658 个，差的 82 个全是标点。
"""
from audio_transcribe.engines.qwen import QwenEngine, to_segments


def test_real_shape_reattaches_punctuation_stripped_from_items():
    """核心坑：items 不含标点，text 含标点。单段场景下标点要原样贴回原位。"""
    raw = {"text": "文庆生，",
           "items": [{"text": "文", "start": 57.36, "end": 57.52},
                     {"text": "庆", "start": 57.52, "end": 57.68},
                     {"text": "生", "start": 57.68, "end": 58.0}]}
    segs = to_segments(raw)
    assert len(segs) == 1
    s = segs[0]
    assert s["text"] == "文庆生，"                                 # 标点回来了
    assert [w["word"] for w in s["words"]] == ["文", "庆", "生"]    # words 里没有标点
    assert s["start"] == 57.36 and s["end"] == 58.0


def test_real_shape_gap_splits_and_each_side_keeps_its_own_punctuation():
    """真机实测「立」到「文」之间有 3.44 秒停顿（>= 默认 800ms 阈值），应切成两段；
    句号跟着刚结束的那段、逗号跟着各自段，不会串到对面。"""
    raw = {"text": "大中，请起立。文庆生，",
           "items": [{"text": "大", "start": 53.2, "end": 53.28},
                     {"text": "中", "start": 53.28, "end": 53.44},
                     {"text": "请", "start": 53.44, "end": 53.68},
                     {"text": "起", "start": 53.68, "end": 53.92},
                     {"text": "立", "start": 53.92, "end": 53.92},
                     {"text": "文", "start": 57.36, "end": 57.52},
                     {"text": "庆", "start": 57.52, "end": 57.68},
                     {"text": "生", "start": 57.68, "end": 58.0}]}
    segs = to_segments(raw)
    assert [s["text"] for s in segs] == ["大中，请起立。", "文庆生，"]
    assert segs[0]["start"] == 53.2 and segs[0]["end"] == 53.92
    assert segs[1]["start"] == 57.36 and segs[1]["end"] == 58.0


def test_zero_duration_item_does_not_break_grouping():
    """真机实测「立」字 start==end==53.92（forced aligner 在停顿边界偶尔给零时长，
    5 分钟样本 658 个 item 里有 20 个），分段逻辑必须容忍，不能除零/崩溃。"""
    raw = {"text": "起立",
           "items": [{"text": "起", "start": 53.68, "end": 53.92},
                     {"text": "立", "start": 53.92, "end": 53.92}]}
    segs = to_segments(raw)
    assert len(segs) == 1
    assert segs[0]["text"] == "起立"
    assert segs[0]["words"][-1]["start"] == segs[0]["words"][-1]["end"] == 53.92


def test_no_timestamps_degrades_to_single_whole_text_segment():
    """真机实测 return_time_stamps=False 时 time_stamps 是 None（QwenEngine 转成
    items=[]）。降级为整段，start=0，end 用调用方传入的音频总时长。"""
    segs = to_segments({"text": "整段话，没有时间戳。", "items": []}, duration=12.5)
    assert segs == [{"start": 0.0, "end": 12.5, "text": "整段话，没有时间戳。",
                      "avg_logprob": 0.0, "no_speech_prob": 0.0, "words": []}]


def test_confidence_defaults_do_not_trip_hallucination_filter():
    """Qwen 不产出置信度。默认值必须是高置信（0.0），
    否则会被幻觉判据 avg_logprob < -1.0 整段误杀（同 FunASR 的取舍）。"""
    raw = {"text": "甲", "items": [{"text": "甲", "start": 0.0, "end": 0.2}]}
    s = to_segments(raw)[0]
    assert s["avg_logprob"] >= -1.0
    assert s["no_speech_prob"] < 0.6


def test_empty_and_missing_fields_are_tolerated():
    assert to_segments({}) == []
    assert to_segments({"text": "", "items": []}) == []
    assert to_segments({"text": None, "items": None}) == []
    # items 非空但 text 是空串：找不到任何 item 文本，走对齐失败的整体降级路径
    assert to_segments({"text": "", "items": [{"text": "甲", "start": 0, "end": 1}]}) == []


def test_overlong_run_is_split_at_max_duration():
    """连续无停顿的长音频（比如慢速念诵）不许整篇一段——下游按 30 秒桶合并，
    超长段会跨桶捣乱。纯内容无标点场景，单独验证切分不丢字（同 FunASR 的对应测试）。"""
    n = 300
    text = "字" * n
    items = [{"text": "字", "start": i * 0.2, "end": i * 0.2 + 0.18} for i in range(n)]
    segs = to_segments({"text": text, "items": items}, max_dur_s=28.0)
    assert len(segs) >= 2
    assert all(s["end"] - s["start"] <= 28.5 for s in segs)
    assert "".join(s["text"] for s in segs) == text


def test_mismatched_items_degrade_to_single_segment_not_crash():
    """items 里出现 text 里根本找不到的内容（对齐假设被打破）时整体降级：
    退回整段原文、words 置空，不猜、不崩。"""
    raw = {"text": "甲乙丙",
           "items": [{"text": "甲", "start": 0.0, "end": 0.2},
                     {"text": "完全对不上的词", "start": 1.0, "end": 2.0}]}
    segs = to_segments(raw)
    assert len(segs) == 1
    assert segs[0]["text"] == "甲乙丙"
    assert segs[0]["words"] == []


def test_qwen_engine_is_registered_and_constructible_without_the_dependency():
    """引擎类必须在没装 qwen-asr/torch 的环境里也能 import 与实例化——重依赖只在
    transcribe() 内部延迟加载，CI 没 GPU、没装 qwen-asr 也要能跑到这一步（见 pyproject.toml
    的 CI 注释：ASR 后端不进核心/dev 依赖）。"""
    from audio_transcribe.engines import Engine, get_engine
    e = get_engine("qwen")
    assert isinstance(e, QwenEngine)
    assert isinstance(e, Engine)
    assert e.name == "qwen"
    assert e.language == "Chinese"
