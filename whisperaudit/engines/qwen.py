"""Qwen3-ASR + Qwen3-ForcedAligner 第三引擎。

实测输出格式（qwen-asr 0.0.6，2026-08-06 真机探测，RTX 4060 8GB smoke.wav 5 分钟中文
语音，不是照文档抄的）：

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda"),
        dtype=torch.bfloat16, device_map="cuda")
    r = model.transcribe(wav, language="Chinese", return_time_stamps=True)[0]
    # r: ASRTranscription(language: str, text: str, time_stamps: ForcedAlignResult | None)
    r.text                 # 完整正文，模型自带标点："大中，请起立。文庆生，…"
    r.time_stamps.items    # [ForcedAlignItem(text="大", start_time=53.2, end_time=53.28), …]
                            # start/end 已经是秒（float），不是毫秒——别再 /1000（FunASR 的坑）

⚠️ 最大的坑：`r.time_stamps.items` **不等于** `r.text` 按字拆开。
Qwen3-ForcedAligner 的分词器（`Qwen3ForceAlignProcessor.clean_token`）只保留字母数字，
标点全部丢弃——实测 5 分钟音频 `r.text` 740 字，`items` 只有 658 个（少了 82 个标点字符）。
拿 items 直接 zip 回 text，或照抄 FunASR「数量对不上就整条降级」的判据（这里数量永远
对不上，会导致这个引擎的时间戳功能形同虚设），都是错的。`to_segments()` 用游标把
items 依次在 text 里定位（`str.find`），标点原样贴回紧邻的前一个 item 所在的分段——
这样切出来的每段既有准确时间戳，又不丢 Qwen 自带的标点。真拼不上（items 出现 text
里根本没有的内容）才整体降级：退回整段原文、words 置空，不猜、不崩。

其它实测坑：
- 自动分块，且**带不带时间戳分块阈值不同**：`return_time_stamps=True` 时按 180 秒
  （`MAX_FORCE_ALIGN_INPUT_SECONDS`）分块，`=False` 时按 1200 秒（`MAX_ASR_INPUT_SECONDS`）。
  库自己做偏移校正和拼接，同一份 5 分钟音频两种调用**文本不完全相同**（740 字 vs 646 字）——
  这是模型在不同分块下重新生成的正常差异，不是 bug，但意味着别指望两种调用逐字一致。
  实测 180 秒分块处（约 176~180s）内容连贯，未见丢字/重复，但只验证过这一个边界。
- 静音会幻觉：1 秒纯静音喂进去吐 `"嗯。"`（items=[{"嗯", 0.0, 0.0}]），不是空字符串。
  engine 层不做幻觉过滤，指望下游 audit.py 的 avg_logprob/字符密度判据兜底——但见下一条，
  这个引擎给不出真置信度，静音幻觉目前没有客观信号能拦。
- items 里约 3%（5 分钟样本 658 条里 20 条）`start == end`（零时长），是 forced aligner
  内部时间戳平滑（`fix_timestamp` 的 LIS 修复）在停顿边界的正常产物，分段逻辑必须容忍。
- 模型不产出置信度分数（没有 avg_logprob/no_speech_prob 这类字段）——按 FunASR 的先例，
  默认填 0.0（高置信），不要填负数，否则会被 audit.py 的幻觉判据 `avg_logprob < -1.0`
  误杀；代价是这个引擎的输出完全绕开幻觉审计。
- 显存：两个 0.6B 模型 bf16 常驻，加载后 allocated ~3.4GB；5 分钟音频转录峰值实测
  reserved 6.72GB／8GB（RTX 4060 Laptop）——8GB 卡上跑得动，但余量不算宽裕。
- 首次加载要下载两个模型（各 1.8GB，共 3.6GB）。经代理下载在这台机器上实测遇到
  **hf_xet 断点续传卡死**：同一字节偏移量（约 1.27GB 处）两次独立重试都卡住不动、
  进程本身没退出也没报错，纯粹不再收数据。设环境变量 `HF_HUB_DISABLE_XET=1` 改走
  huggingface_hub 传统 HTTP 下载后恢复正常（期间仍会零星 `Read timed out`，
  但这条路径自带自动重试续传，扛得住）。
- 速度（RTX 4060 8GB，warm cache，5 分钟音频）：加载（两个模型一起上卡）20.7 秒；
  推理 return_time_stamps=True 约 18~19x 实时，=False（不跑 forced aligner）约 24.8x 实时。
"""
from . import Engine, register


def _group(items, gap_ms=800, max_dur_s=28.0):
    """按相邻 item 的间隙分组，组也不许超过 max_dur_s——连续无停顿的长音频（比如慢速
    念诵）不许整篇一段，下游按 30 秒桶合并，超长段会跨桶捣乱（同 FunASR 的取舍）。

    items 须按时间升序排列：[{"text", "start", "end"}, …]。"""
    groups, cur = [], []
    for it in items:
        if cur and (it["start"] - cur[-1]["end"] >= gap_ms / 1000
                    or it["start"] - cur[0]["start"] >= max_dur_s):
            groups.append(cur)
            cur = []
        cur.append(it)
    if cur:
        groups.append(cur)
    return groups


def _whole(text, duration):
    """整体降级：拼不出可信的分段，退回整段原文、words 置空，下游自动退化
    （字幕退回整段、段内标点不加——但 Qwen 的 text 本来就带标点，退化代价比 FunASR 小）。"""
    if not text:
        return []
    return [{"start": 0.0, "end": float(duration), "text": text,
             "avg_logprob": 0.0, "no_speech_prob": 0.0, "words": []}]


def to_segments(raw, duration=0.0, gap_ms=800, max_dur_s=28.0):
    """把 Qwen3-ASR + ForcedAligner 的输出转成流水线的 segment 形状。纯函数，可离线测试。

    raw 形状（QwenEngine.transcribe() 里组装，来自真机探测）：
        {"text": "大中，请起立。…",                          # 完整正文，模型自带标点
         "items": [{"text": "大", "start": 53.2, "end": 53.28}, …]}  # 秒，非毫秒

    items 不等于 text 按字拆开——见模块 docstring。这里把 items 依次在 text 里定位，
    标点原样贴回紧邻的前一个 item 所在的分段；对不上时整体降级（见 `_whole`）。

    duration：无时间戳（items 为空但 text 非空）时唯一整段的 end，调用方传入音频总时长。
    """
    text = raw.get("text") or ""
    items = raw.get("items") or []
    if not items:
        return _whole(text, duration)

    groups = _group(items, gap_ms=gap_ms, max_dur_s=max_dur_s)
    flat = [(gi, it) for gi, grp in enumerate(groups) for it in grp]
    seg_texts = ["" for _ in groups]
    cursor = 0
    for gi, it in flat:
        w = it["text"]
        pos = text.find(w, cursor) if w else cursor
        if pos < 0:                                          # 对齐假设被打破
            return _whole(text, duration)
        if pos > cursor:                                      # 中间隔着标点/空白
            target = gi if seg_texts[gi] else max(gi - 1, 0)   # 贴给刚结束的那段
            seg_texts[target] += text[cursor:pos]
        seg_texts[gi] += w
        cursor = pos + len(w)
    if cursor < len(text):                                    # 收尾的标点
        seg_texts[-1] += text[cursor:]

    return [{"start": grp[0]["start"], "end": grp[-1]["end"], "text": txt,
             # Qwen 不产出置信度；置 0.0（高置信）而不是负数，
             # 免得被幻觉判据 avg_logprob < -1.0 误杀（同 FunASR 的取舍）
             "avg_logprob": 0.0, "no_speech_prob": 0.0,
             "words": [{"start": w["start"], "end": w["end"], "word": w["text"]}
                       for w in grp]}
            for grp, txt in zip(groups, seg_texts)]


def _duration(wav):
    """音频时长。标准库 wave 只认 int16 PCM，float32 WAV 会直接抛错——
    用 soundfile 兜住，读不出来就返回 0.0，调用方按 segments 最后一段兜底
    （同 FunASR 的 _duration，但签名更简单：Qwen 这条路的兜底顺序反过来更自然）。"""
    try:
        import soundfile
        return float(soundfile.info(wav).duration)
    except Exception:
        return 0.0


@register
class QwenEngine(Engine):
    name = "qwen"

    def __init__(self, model="Qwen/Qwen3-ASR-0.6B",
                 forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                 device="cuda", dtype="bfloat16", language="Chinese", **_):
        self.model_name = model
        self.forced_aligner_name = forced_aligner
        self.device = device
        self.dtype_name = dtype
        self.language = language
        self._model = None

    def transcribe(self, wav, language=None, return_time_stamps=True, **_):
        import torch
        from qwen_asr import Qwen3ASRModel

        if self._model is None:
            dtype = getattr(torch, self.dtype_name)
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                forced_aligner=self.forced_aligner_name,
                forced_aligner_kwargs=dict(dtype=dtype, device_map=self.device),
                dtype=dtype, device_map=self.device)

        results = self._model.transcribe(
            wav, language=language or self.language, return_time_stamps=return_time_stamps)
        r = results[0]
        items = []
        if return_time_stamps and r.time_stamps is not None:
            items = [{"text": it.text, "start": it.start_time, "end": it.end_time}
                      for it in r.time_stamps.items]

        dur = _duration(wav)
        segs = to_segments({"text": r.text, "items": items}, duration=dur)
        if not dur:
            dur = segs[-1]["end"] if segs else 0.0
        return {"duration": dur, "segments": segs}
