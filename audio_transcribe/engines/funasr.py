"""FunASR（SeacoParaformer）第二引擎。

实测输出格式（funasr 1.1.16，2026-08-06 真机探测，不是照文档抄的）：

    [{"key": "音频名",
      "text": "转 出 的 每 个 字 …",            # 字间带空格
      "timestamp": [[53150, 53330], …]}]     # 逐字 [start_ms, end_ms]，与字一一对应

没有句级切分（sentence_timestamp 在本配置下不生效），但逐字时间戳比句级更好：
适配器按字间停顿切段，顺手就把词级时间戳（words）构造出来了——
下游的字幕重切、段内标点、饥饿检测全都直接可用。

环境坑（都实测踩过，记在 docs/measurements.md）：
- funasr 1.4.x 的 `paraformer-zh` 别名解析不到类，1.1.16 同样；必须传**本地模型目录**
- ModelScope 的 seaco 仓库缺 `seg_dict`（404），要从同 vocab 的非 seaco 仓库单独拉一份
"""
from . import Engine, register

MODEL_DIR = ("~/.cache/modelscope/hub/models/iic/"
             "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")


def to_segments(raw, gap_ms=800, max_dur_s=28.0):
    """把 FunASR 原始输出转成流水线的 segment 形状。纯函数，可离线测试。

    按相邻字的间隙 >= gap_ms 切段，段也不许超过 max_dur_s（防止一路慢速歌唱
    连绵不断导致整篇一段——下游按 30 秒桶合并，超长段会跨桶捣乱）。

    text 与 timestamp 数量对不上时整条降级：拼成一段、words 置空，
    下游自动退化（字幕退回整段、段内标点不加），不许崩。
    """
    out = []
    for item in raw:
        chars = (item.get("text") or "").split()
        ts = item.get("timestamp") or []
        if not chars:
            continue
        if len(chars) != len(ts):
            out.append({"start": ts[0][0] / 1000 if ts else 0.0,
                        "end": ts[-1][1] / 1000 if ts else 0.0,
                        "text": "".join(chars), "avg_logprob": 0.0,
                        "no_speech_prob": 0.0, "words": []})
            continue
        seg = []
        for ch, (a, b) in zip(chars, ts):
            w = {"start": a / 1000, "end": b / 1000, "word": ch}
            if seg and (a / 1000 - seg[-1]["end"] >= gap_ms / 1000
                        or a / 1000 - seg[0]["start"] >= max_dur_s):
                out.append(_seg(seg))
                seg = []
            seg.append(w)
        if seg:
            out.append(_seg(seg))
    return out


def _seg(words):
    return {"start": words[0]["start"], "end": words[-1]["end"],
            "text": "".join(w["word"] for w in words),
            # FunASR 不给置信度；置 0.0（高置信）而不是负数，
            # 免得被幻觉判据的 avg_logprob < -1.0 误杀
            "avg_logprob": 0.0, "no_speech_prob": 0.0,
            "words": [dict(w) for w in words]}


@register
class FunASREngine(Engine):
    name = "funasr"

    def __init__(self, model_dir=None, device="cuda", **_):
        import os
        self.model_dir = os.path.expanduser(model_dir or MODEL_DIR)
        self.device = device
        self._model = None

    def transcribe(self, wav, **_):
        from funasr import AutoModel
        if self._model is None:
            self._model = AutoModel(model=self.model_dir, device=self.device,
                                    disable_update=True)
        raw = self._model.generate(input=wav, batch_size_s=300)
        segs = to_segments(raw)
        return {"duration": _duration(wav, segs), "segments": segs}


def _duration(wav, segs):
    """音频时长。标准库 wave 只认 int16 PCM，FLEURS 这类 float32 WAV
    （format 3）会直接抛错——用 soundfile（funasr 依赖链自带）兜住，
    再不行退化到最后一段的结束时间。"""
    try:
        import soundfile
        return float(soundfile.info(wav).duration)
    except Exception:
        return segs[-1]["end"] if segs else 0.0
