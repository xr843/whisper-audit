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

MODEL_ID = ("iic/"
            "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

# 整条喂给 Paraformer 的音频不能超过这个长度——注意力显存随长度**平方**增长。
# 2026-08-07 实测（8GB 卡）：15 分钟能跑，34 分钟要 33GiB、48 分钟要 35GiB，
# 直接 OOM（或先撞 cuDNN NOT_SUPPORTED）。而此前所有冒烟都在 5~15 分钟音频上做，
# 「长音频项目的推荐引擎吃不下长音频」这个事实是三个困难域评测同时崩掉才暴露的。
# 取 300 秒：远在实测安全线内，且单窗前向本来就快，分窗几乎无速度代价。
MAX_CHUNK_S = 300.0
# 窗边界在名义位置 ±3 秒内挪到最安静处，避免把字从中间切开。
_SEARCH_S = 3.0


def split_points(audio, sr, max_chunk_s=MAX_CHUNK_S, search_s=_SEARCH_S):
    """长音频分窗的切点（样本数），首尾必含 [0, len]。纯函数，可离线测试。

    每个名义边界（i * max_chunk_s）在 ±search_s 范围内滑一个 200ms 窗找
    RMS 最小处，把切点放在那个窗的中心——句间停顿几乎总能被找到；
    整段都响（无停顿）时退化为切在名义位置附近，不多想：Paraformer 对
    截断字的代价是错一两个字，远小于 OOM 崩掉整条。
    """
    n = len(audio)
    max_len = int(max_chunk_s * sr)
    pts = [0]
    while n - pts[-1] > max_len:
        nominal = pts[-1] + max_len
        win, step = int(0.2 * sr), int(0.05 * sr)
        lo = max(pts[-1] + win, nominal - int(search_s * sr))
        hi = min(n - win, nominal + int(search_s * sr))
        best, best_rms = nominal, float("inf")
        for s in range(lo, hi - win, step):
            seg = audio[s:s + win].astype("float64")
            rms = float((seg * seg).mean())
            if rms < best_rms:
                best_rms, best = rms, s + win // 2
        pts.append(best)
    pts.append(n)
    return pts


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
        # 显式给了目录就信它；默认目录推迟到首次转录时 ensure_model 解析——
        # 构造必须离线零成本（get_engine 在 CI 里就会跑到这儿）。
        self.model_dir = os.path.expanduser(model_dir) if model_dir else None
        self.device = device
        self._model = None

    def transcribe(self, wav, **_):
        import os
        import tempfile

        import soundfile as sf

        from funasr import AutoModel
        if self._model is None:
            if self.model_dir is None:
                from . import ensure_model
                self.model_dir = ensure_model(MODEL_ID,
                                              require=("model.pt", "seg_dict"))
            self._model = AutoModel(model=self.model_dir, device=self.device,
                                    disable_update=True)

        # 短音频保持原样直接喂路径——这是被真机探测验证过的调用形式，不动它。
        if sf.info(wav).duration <= MAX_CHUNK_S:
            segs = to_segments(self._model.generate(input=wav, batch_size_s=300))
            return {"duration": _duration(wav, segs), "segments": segs}

        # 长音频分窗。窗内仍走「写临时 wav、喂路径」的同一条已验证路径，
        # 不改用数组输入——那是另一个没探测过的 API 形态。
        audio, sr = sf.read(wav, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        pts = split_points(audio, sr)
        segs = []
        with tempfile.TemporaryDirectory() as td:
            for i, (a, b) in enumerate(zip(pts, pts[1:])):
                p = os.path.join(td, f"chunk{i:03d}.wav")
                sf.write(p, audio[a:b], sr, subtype="PCM_16")
                off = a / sr
                for s in to_segments(self._model.generate(input=p, batch_size_s=300)):
                    s["start"] += off
                    s["end"] += off
                    for w in s["words"]:
                        w["start"] += off
                        w["end"] += off
                    segs.append(s)
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
