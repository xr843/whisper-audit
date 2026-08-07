"""faster-whisper 引擎。"""
import json
import os
import subprocess
import time

from .. import log
from . import Engine, register


# ---------------------------------------------------------------- 转录

def transcribe_pass(wav, out_json, chunk_length, batch, beam, compute="float16",
                    model_name="large-v3", device="cuda", language="zh"):
    """一路转录。

    刻意不传 initial_prompt：实测模型会在静音段把 prompt 原样"转录"出来，
    还占掉真实语音的时间段（有个案例吞了 88 秒）。术语引导交给事后的术语表。

    开 word_timestamps：段级时间戳被 VAD 拉得很宽（最长一段 296 秒），
    拿它做字幕和段内标点都不成立，必须要词级的。"""
    if out_json and os.path.exists(out_json):
        log(f"复用已有 {os.path.basename(out_json)}")
        return json.load(open(out_json, encoding="utf-8"))

    from faster_whisper import WhisperModel, BatchedInferencePipeline
    kw = {}
    if chunk_length:
        kw["chunk_length"] = chunk_length

    rows, info, el = None, None, None
    for attempt_batch in (batch, batch // 2, max(1, batch // 4)):
        log(f"转录 chunk={chunk_length}s batch={attempt_batch} beam={beam} ({compute}) …")
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute)
            pipe = BatchedInferencePipeline(model=model)
            t0 = time.time()
            segs, info = pipe.transcribe(
                wav, language=language, batch_size=attempt_batch, beam_size=beam,
                vad_filter=True, word_timestamps=True,
                vad_parameters=dict(min_silence_duration_ms=500 if chunk_length >= 30 else 200,
                                    speech_pad_ms=400 if chunk_length >= 30 else 200),
                condition_on_previous_text=False,      # 防止幻觉沿着上下文传播
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                no_speech_threshold=0.6,
                **kw)
            rows, last = [], 0.0
            for s in segs:
                rows.append({"start": round(s.start, 3), "end": round(s.end, 3),
                             "text": s.text.strip(),
                             "avg_logprob": round(s.avg_logprob, 4),
                             "no_speech_prob": round(s.no_speech_prob, 4),
                             "words": [{"start": round(w.start, 3), "end": round(w.end, 3),
                                        "word": w.word}
                                       for w in (getattr(s, "words", None) or [])]})
                if s.end - last >= 600:
                    last = s.end
                    el = time.time() - t0
                    log(f"  {100*s.end/info.duration:5.1f}%  {s.end/max(el,1e-9):.1f}x")
            el = time.time() - t0
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or attempt_batch == max(1, batch // 4):
                raise
            log(f"  显存不足（batch={attempt_batch}），降档重试")
            try:
                del pipe, model
            except NameError:
                pass
            import gc
            gc.collect()
    if rows is None:
        raise RuntimeError("转录失败")
    log(f"  完成 {len(rows)} 段 / {sum(len(r['text']) for r in rows):,} 字 / "
        f"{el/60:.1f}min / {info.duration/el:.1f}x")
    d = {"duration": info.duration, "segments": rows}
    if out_json:
        json.dump(d, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return d


# ---------------------------------------------------------------- 补转

def repatch(wav, spans, out_json, pad=6.0, model_name="large-v3", device="cuda",
            compute="float16", language="zh"):
    """只对问题区段重转。关 VAD（它已经判错一次了）、抬高 no_speech 门槛、给足上下文。"""
    if not spans:
        return []
    if out_json and os.path.exists(out_json):
        log(f"复用已有 {os.path.basename(out_json)}")
        return json.load(open(out_json, encoding="utf-8"))
    from faster_whisper import WhisperModel
    log(f"定点补转 {len(spans)} 处（{sum(b-a for a,b,_ in spans):.0f} 秒）…")
    model = WhisperModel(model_name, device=device, compute_type=compute)
    tmpdir = os.path.dirname(os.path.abspath(out_json))
    res = []
    for i, (a, b, label) in enumerate(spans):
        a2, b2 = max(0, a - pad), b + pad
        tmp = os.path.join(tmpdir, f"_span_{i:03d}.wav")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", wav,
                        "-ss", str(a2), "-to", str(b2), "-c", "copy", tmp], check=True)
        attempts = []
        for mode, kw in (("novad", dict(vad_filter=False)),
                         ("vad_loose", dict(vad_filter=True,
                                            vad_parameters=dict(min_silence_duration_ms=2000,
                                                                threshold=0.2, speech_pad_ms=800)))):
            segs, _ = model.transcribe(tmp, language=language, beam_size=5,
                                       condition_on_previous_text=False,
                                       temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                       no_speech_threshold=0.9,   # 宁可多转，不可漏转
                                       log_prob_threshold=-2.0,
                                       word_timestamps=True, **kw)
            attempts.append({"mode": mode, "rows": [
                {"start": round(a2 + s.start, 2), "end": round(a2 + s.end, 2),
                 "text": s.text.strip(),
                 "words": [{"start": round(a2 + w.start, 3), "end": round(a2 + w.end, 3),
                            "word": w.word}
                           for w in (getattr(s, "words", None) or [])]}
                for s in segs]})
        os.remove(tmp)
        res.append({"span": [a, b], "label": label, "attempts": attempts})
    json.dump(res, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


@register
class WhisperEngine(Engine):
    name = "whisper"

    def __init__(self, model_name="large-v3", device="cuda", compute="int8_float16"):
        self.model_name, self.device, self.compute = model_name, device, compute

    def transcribe(self, wav, out_json=None, chunk_length=30, batch=16, beam=5,
                   language="zh", **_):
        return transcribe_pass(wav, out_json, chunk_length, batch, beam,
                               self.compute, self.model_name, self.device, language)
