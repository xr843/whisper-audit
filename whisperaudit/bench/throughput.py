#!/usr/bin/env python3
"""默认档吞吐基准。

基线必须可复现，所以固定一段基准音频测吞吐，而不是拿「上次那份录音跑了多久」
当基线——那份录音的源音频已丢失，无法复跑。

基线测的是**流水线实际跑的配置**（词级时间戳开）。拿关闭时的数字卡门禁
等于在测一个不存在的配置。

⚠️ 跑之前先确认 GPU 是空的（nvidia-smi）。2026-08-05 有一次在 GPU 忙时测出
8.4x 并写进了文档，真值是 24.5x —— 差了三倍。测的是排队情况，不是代码。

⚠️ 这个倍数只用于**同一段音频的回归比较**。bench15.wav 是静音很多的照稿朗读片段，
VAD 砍掉近半，倍数虚高；不要拿它预估语音密集的讲座录音要跑多久。

    python3 bench/throughput.py --wav bench/data/bench15.wav
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisperaudit.audio import ensure_cuda_libs

BASELINE = 24.5        # bench15.wav（15 分钟照稿朗读片段）、int8_float16、batch16、beam5、词级时间戳开
TOLERANCE = 0.05        # 退化不得超过 5% → 下限 23.3x


def run_once(wav, words, model_name, compute, batch, beam):
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    m = WhisperModel(model_name, device="cuda", compute_type=compute)
    pipe = BatchedInferencePipeline(model=m)
    t0 = time.time()
    segs, info = pipe.transcribe(
        wav, language="zh", batch_size=batch, beam_size=beam, chunk_length=30,
        vad_filter=True, word_timestamps=words,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400),
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], no_speech_threshold=0.6)
    # 必须消费生成器——faster-whisper 是惰性的，不消费就等于没转录，计时会假到离谱
    n = sum(len(s.text) for s in segs)
    el = time.time() - t0
    return info.duration / el, n, info.duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    # 默认开——基线必须测流水线实际跑的配置
    ap.add_argument("--no-words", dest="words", action="store_false", default=True,
                    help="关词级时间戳（只用于对比，不是默认档配置）")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--compute", default="int8_float16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--baseline", type=float, default=BASELINE)
    a = ap.parse_args()
    ensure_cuda_libs()

    kw = dict(model_name=a.model, compute=a.compute, batch=a.batch, beam=a.beam)
    print(f"预热…", flush=True)
    _, _, dur = run_once(a.wav, a.words, **kw)
    print(f"音频 {dur/60:.1f} 分钟　词级时间戳 {'开' if a.words else '关'}　"
          f"{a.model}/{a.compute}/batch{a.batch}/beam{a.beam}", flush=True)

    xs = []
    for i in range(a.runs):
        x, n, _ = run_once(a.wav, a.words, **kw)
        print(f"  第 {i+1} 次  {x:5.2f}x  {n:,} 字", flush=True)
        xs.append(x)
    x = statistics.mean(xs)
    floor = a.baseline * (1 - TOLERANCE)
    print(f"\n均值 {x:.2f}x　基线 {a.baseline}x　下限 {floor:.2f}x　"
          f"→ {'通过' if x >= floor else '不通过'}")
    return 0 if x >= floor else 1


if __name__ == "__main__":
    sys.exit(main())
