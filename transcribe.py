#!/usr/bin/env python3
"""长音频转录流水线 —— 以「不遗漏」为目标，而不是「转一遍」。

    python3 transcribe.py 录音.mp3 -o 输出目录 --profile meeting --terms terms/xxx.json

单跑一遍 Whisper 一定会漏。本流水线做四件单遍做不到的事：
  1. 双路转录（chunk=30 与 chunk=10），实测两路互补、各有各的漏，取并集
  2. 覆盖率审计：找出时间轴空洞，并测每个空洞的实际音量——音量接近讲话的空洞就是漏转
  3. 幻觉识别：静音段会被转成网络字幕套语（"请不吝点赞订阅…"），靠音量+关键词+置信度三重判定
  4. 定点补转：只对问题区段重转，3 分钟解决问题，不必再跑一遍全量（那要 84 分钟）

详见 README.md 的「踩过的坑」，每一条都是实测出来的，不是理论。
"""
import argparse, json, math, os, re, subprocess, sys, time, wave


def ensure_cuda_libs():
    """CTranslate2 要在运行时找到 cuBLAS/cuDNN，它们随 torch 的 pip 包装在 nvidia/ 下。
    LD_LIBRARY_PATH 必须在进程启动前生效，所以这里设好后重启自己。

    只能从 main() 调用，绝不能放在模块顶层：import 本模块的进程会被 execv 顶替掉，
    实测这会让 pytest 静默退出（退出码 1、零输出）。"""
    try:
        import nvidia
    except ImportError:
        return
    p = os.path.dirname(nvidia.__file__)
    libs = [os.path.join(p, d, "lib") for d in os.listdir(p)
            if os.path.isdir(os.path.join(p, d, "lib"))]
    if not libs:
        return
    want = ":".join(libs)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if want.split(":")[0] not in cur:
        os.environ["LD_LIBRARY_PATH"] = want + (":" + cur if cur else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------- 参数档

# batch=16 必须配 int8_float16：8GB 卡上 fp16+batch16 会 OOM（实测）。
# int8 本身不提速，它的价值就是省出显存来开大 batch —— 那才是提速来源。
PROFILES = {
    # 单人讲授、音质好：单路 + 补转即可，最快
    "lecture": {"two_pass": False, "chunk_coarse": 30, "batch": 16, "beam": 5,
                "compute": "int8_float16"},
    # 多人问答、口音重、内容重要：双路交叉，最全
    "meeting": {"two_pass": True, "chunk_coarse": 30, "chunk_fine": 10, "batch": 16, "beam": 5,
                "compute": "int8_float16"},
    # 只求快（质量有代价，专业术语多的内容不建议）
    "fast": {"two_pass": False, "chunk_coarse": 30, "batch": 16, "beam": 1,
             "compute": "int8_float16"},
}

# 模型在静音段会吐训练数据里的网络视频字幕套语。这些词出现即幻觉。
HALLU_WORDS = (r"点赞|订阅|打赏|字幕志愿者|中文字幕|谢谢观看|感谢观看|"
               r"请不吝|字幕组|翻译by|本视频|下期再见")
HALLU_PAT = re.compile(HALLU_WORDS)

# 正常汉语讲授的语速。实测这份 3.65 小时讲座的 segment 中位密度是 3.63 字/秒，
# 取 3.0 作保守基准：用来估「有效语音时长」和判定段内饥饿。
SPEECH_RATE = 3.0

# 连接词前不能断句的粘连字。实测 15 处「这是，第一个环节」都是因为
# 只看连接词、不看前一个字——第一个/第二个在汉语里更多是宾语而非句首。
#
# 只收**前置**成分（是第一个／就比如说／是因为）。
# 「的」「了」是句末助词，方向正好相反——它们后面恰恰该断句
# （的，所以说／了，所以说／的，另外），把它们收进来会挡掉 16 个正确的逗号。
CLAUSE_GLUE = "是为把被将对与和及或即含有到就都也很才又更最称叫做算"


# ---------------------------------------------------------------- 工具

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hms(t):
    t = int(t)
    return f"{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}"


def prepare_audio(src, workdir):
    """Whisper 只吃 16kHz 单声道。"""
    wav = os.path.join(workdir, "audio16k.wav")
    if os.path.exists(wav):
        log(f"复用已有 {wav}")
        return wav
    log(f"转码 {os.path.basename(src)} -> 16kHz 单声道")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
    return wav


class Loudness:
    """按时间区间测 RMS 音量。判断空洞是真静音还是漏转，全靠它。"""

    def __init__(self, wav):
        self.w = wave.open(wav, "rb")
        self.sr = self.w.getframerate()
        self.n = self.w.getnframes()

    def db(self, t0, t1):
        import numpy as np
        a0 = max(0, int(t0 * self.sr))
        a1 = min(self.n, int(t1 * self.sr))
        if a1 - a0 < self.sr // 20:
            return float("nan")
        self.w.setpos(a0)
        a = np.frombuffer(self.w.readframes(a1 - a0), dtype=np.int16).astype("float32")
        if a.size == 0:
            return float("nan")
        return 20 * math.log10(max(float((a ** 2).mean()) ** 0.5, 1e-9) / 32768)

    def close(self):
        self.w.close()


# ---------------------------------------------------------------- 转录

def transcribe_pass(wav, out_json, chunk_length, batch, beam, compute="float16",
                    model_name="large-v3", device="cuda", language="zh"):
    """一路转录。

    刻意不传 initial_prompt：实测模型会在静音段把 prompt 原样"转录"出来，
    还占掉真实语音的时间段（有个案例吞了 88 秒）。术语引导交给事后的术语表。

    开 word_timestamps：段级时间戳被 VAD 拉得很宽（最长一段 296 秒），
    拿它做字幕和段内标点都不成立，必须要词级的。"""
    if os.path.exists(out_json):
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
    json.dump(d, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return d


# ---------------------------------------------------------------- 审计

def starved_spans(segments, min_len=15.0, min_density=1.2, max_span=90.0):
    """段内饥饿：一个 segment 时长很长、字却很少 —— 它内部吞掉了没转出来的话。

    这是覆盖率审计最大的盲区。VAD 会剪掉静音再把时间戳映射回原轴，
    于是一个 segment 的时间跨度可以远大于它真正的语音，而 audit() 按首尾
    合并区间算覆盖，这段时间就被算作「已覆盖」。

    实测最坏的一处：1575–1871 秒，296 秒只转出 128 字，avg_logprob 却有 −0.28。
    所以这里**刻意不看置信度** —— 幻觉那两条判据都写成 `low_conf and ...`，
    高置信度直接豁免，恰好放过了损失最大的那一段。

    整段丢给补转会退化成一次全量重跑，故按 max_span 切块。
    """
    out = []
    for s in segments:
        a, b = s["start"], s["end"]
        if b - a < min_len:
            continue                      # 短段交给幻觉那条路，否则补转清单会被碎片淹没
        if len(s["text"].strip()) / max(b - a, 1e-9) >= min_density:
            continue
        n = max(1, int(math.ceil((b - a) / max_span)))
        step = (b - a) / n
        for k in range(n):
            out.append([round(a + k * step, 1), round(a + (k + 1) * step, 1), "段内饥饿"])
    return out


def audit_rows(rows, dur, min_len=15.0, min_density=1.2):
    """对**最终合并稿**做审计——不需要音频，纯看时间轴与字符密度。

    过去质检报告审的是 pass1，交付的却是合并稿：实测 pass1 报 97.3%，
    而成品字幕的时间并集只有 90.0%。给读者看的数字必须来自交付物本身。
    """
    spans = sorted(([r["start"], r["end"]] for r in rows), key=lambda x: x[0])
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cover = sum(b - a for a, b in merged)
    # 有效语音：一段最多只能算它的字数撑得起的那点时长
    speech = sum(min(r["end"] - r["start"], len(r["text"].strip()) / SPEECH_RATE)
                 for r in rows)
    starved = sum(1 for r in rows
                  if r["end"] - r["start"] >= min_len
                  and len(r["text"].strip()) / max(r["end"] - r["start"], 1e-9) < min_density)
    return {"duration": dur, "cover": cover,
            "cover_pct": 100 * cover / dur if dur else 0.0,
            "speech": speech, "speech_pct": 100 * speech / dur if dur else 0.0,
            "starved": starved, "chars": sum(len(r["text"]) for r in rows)}


def audit(data, loud, gap_min=3.0):
    """找空洞、量音量、揪幻觉。返回需要补转的区段清单。"""
    segs = sorted(data["segments"], key=lambda x: x["start"])
    dur = data["duration"]
    merged = []
    for s in segs:
        if merged and s["start"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s["end"])
        else:
            merged.append([s["start"], s["end"]])
    cover = sum(b - a for a, b in merged)

    speech_db = sorted(loud.db(a, min(b, a + 20)) for a, b in merged[:200] if b - a > 3)
    speech_db = speech_db[len(speech_db) // 2] if speech_db else -30.0

    gaps, prev = [], 0.0
    for a, b in merged:
        if a - prev >= gap_min:
            gaps.append((prev, a))
        prev = max(prev, b)
    if dur - prev >= gap_min:
        gaps.append((prev, dur))

    spans = []
    for a, b in gaps:
        db = loud.db(a, b)
        if db == db and db > speech_db - 8:      # 音量接近讲话 = 有人在说，却没转出来
            spans.append([round(a, 1), round(b, 1), "VAD响空洞"])

    # 幻觉分两类，处理方式相反——判据是音量，不是文本：
    #   音量正常 => 真语音被顶替，内容丢了，要补转
    #   音量偏低 => 那里本来就没内容，要剔除
    hallu, drop = [], []
    for s in segs:
        by_word = bool(HALLU_PAT.search(s["text"]))
        low_conf = s.get("avg_logprob", 0.0) < -1.0
        span_len = s["end"] - s["start"]
        # 说了半天没几个字：正常讲话约 3 字/秒，"嗯嗯嗯"占 28 秒只转出 3 个字。
        starved = (low_conf and span_len > 5
                   and len(s["text"].strip()) / max(span_len, 0.1) < 0.5)
        if not (by_word or low_conf):
            continue
        db = loud.db(s["start"], s["end"])
        if db != db:
            continue
        if starved:                          # 无论音量如何，这种段落没有内容可言
            hallu.append((s, db))
            drop.append([round(s["start"], 1), round(s["end"], 1)])
        elif db > speech_db - 8:
            if by_word:                      # 低置信但音量正常，可能只是难识别，不算幻觉
                hallu.append((s, db))
                spans.append([round(s["start"], 1), round(s["end"], 1), "幻觉替换段"])
        else:
            hallu.append((s, db))
            drop.append([round(s["start"], 1), round(s["end"], 1)])

    spans.extend(starved_spans(segs))
    spans.sort()
    return {"duration": dur, "cover": cover, "cover_pct": 100 * cover / dur,
            "speech_db": speech_db, "gaps": gaps, "spans": spans, "drop": drop,
            "hallu": [(s["start"], s["end"], s["text"], db) for s, db in hallu]}


def find_breaks(data, loud, min_len=120, step=10, max_n=3):
    """找中场休息，返回**全部**符合条件的区段（全天课上下午各一次，只取最长的会漏）。

    判据必须是「音量低」**且**「转出的是幻觉或极少字」——只看音量会两头判偏：
    实测有一版把休息段末尾多划了 47 秒，把"接下来我们开始互动的环节"整句删掉了。
    步长取 10 秒而非 30 秒，边界才够准。
    """
    dur = data["duration"]
    segs = data["segments"]

    dbs = [(t, loud.db(t, t + step)) for t in range(0, int(dur), step)]
    vals = sorted(d for _, d in dbs if d == d)
    if not vals:
        return []
    ref = vals[len(vals) * 3 // 4]           # 上四分位当讲话基准

    def spoken_chars(t0, t1):
        """该窗口里的"真内容"字数——幻觉不算数。

        光靠关键词表认不全：休息段还会冒出"鲍鱼""这间餐厅有很多不同的食物"
        这类跟主题毫不相干的臆造。它们的共同客观特征是 avg_logprob 极低。"""
        n = 0
        for s in segs:
            if not (t0 <= s["start"] < t1):
                continue
            if HALLU_PAT.search(s["text"]):
                continue
            if s.get("avg_logprob", 0.0) < -1.0:      # 低置信 = 大概率臆造
                continue
            n += len(s["text"].strip())
        return n

    flags = []
    for t, d in dbs:
        quiet = (d == d) and d < ref - 6
        flags.append(quiet and spoken_chars(t, t + step) < 8)

    runs, cur = [], None
    for i, f in enumerate(flags):
        if f:
            cur = (dbs[i][0], dbs[i][0] + step) if cur is None else (cur[0], dbs[i][0] + step)
        else:
            if cur:
                runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)

    runs = [r for r in runs if r[1] - r[0] >= min_len]
    runs.sort(key=lambda r: -(r[1] - r[0]))
    return sorted(runs[:max_n])


def in_any(t, spans, pad=0.0):
    return any(a - pad <= t <= b + pad for a, b in spans)


# ---------------------------------------------------------------- 补转

def repatch(wav, spans, out_json, pad=6.0, model_name="large-v3", device="cuda",
            compute="float16", language="zh"):
    """只对问题区段重转。关 VAD（它已经判错一次了）、抬高 no_speech 门槛、给足上下文。"""
    if not spans:
        return []
    if os.path.exists(out_json):
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


# ---------------------------------------------------------------- 合并

def strip_common(a, b, min_block=8):
    """从 b 里删掉与 a 重复的长字块，保留 b 独有的部分。
    整段删是不行的——两路在同一时间窗常各有对方漏掉的新内容。"""
    from difflib import SequenceMatcher
    sm = SequenceMatcher(None, a, b, autojunk=False)
    keep, last = [], 0
    for blk in sm.get_matching_blocks():
        if blk.size >= min_block:
            keep.append(b[last:blk.b])
            last = blk.b + blk.size
    keep.append(b[last:])
    return "".join(keep)


def merge_rows(rows):
    rows = sorted(rows, key=lambda r: (r["start"], -(r["end"] - r["start"])))
    out = []
    for r in rows:
        cur = r["text"]
        for prev in reversed(out[-6:]):
            if prev["end"] > r["start"] + 0.3:
                cur = strip_common(prev["text"], cur, 8)      # 时间重叠 = 转的同一段音频
            else:
                # 时间不重叠：法规名、术语本就会被反复引用，不能按字块裁，
                # 只有整段几乎全重复时才判定是 ASR 重复。
                #
                # 判据必须同时看**绝对**剩余量：法规名越长，剥掉它之后的比例就越小，
                # 光看比例会把后面那句真话一起丢掉。实测
                # 「根据大型活动场所财务管理办法第九条」占 17 字，剩下的
                # 「还要做年度报告」7 字 < 24×0.3=7.2，整行就没了。
                rest = strip_common(prev["text"], cur, 10)
                if len(rest) < 6 and len(rest) < len(cur) * 0.3:
                    cur = ""
            if len(cur.strip()) < 3:
                break
        cur = cur.strip()
        if len(cur) >= 3:
            row = {**r, "text": cur}
            if cur != r["text"].strip():
                # 文本被裁过，词表和它就对不上了。宁可退回整段字幕，
                # 也不要拿错位的词级时间戳去切——那比不切更误导。
                row.pop("words", None)
            out.append(row)
    return out


def combine(passes, patch, terms, breaks, dur, drop_spans=()):
    """逐 30 秒窗口在各路之间取字数多的，再用补转填两路都空的洞。"""
    import opencc
    cc = opencc.OpenCC("t2s")
    fixes = terms.get("fixes", [])

    def norm(t):
        t = cc.convert(t).strip()
        for a, b in fixes:
            t = t.replace(a, b)
        return t

    def usable(t, start):
        if not t or HALLU_PAT.search(t):
            return False
        if in_any(start, breaks):
            return False
        # 落在休息段之外的零散幻觉（音量低+低置信），也要剔除
        if in_any(start, drop_spans, pad=0.2):
            return False
        return True

    prepped = []
    for p in passes:
        prepped.append([{"start": s["start"], "end": s["end"], "text": norm(s["text"]),
                         "words": s.get("words") or []}
                        for s in p["segments"] if usable(norm(s["text"]), s["start"])])

    W = 30.0
    buckets = []
    for rows in prepped:
        h = {}
        for r in rows:
            h.setdefault(int(r["start"] // W), []).append(r)
        buckets.append(h)

    picked = []
    for k in range(int(dur // W) + 2):
        cands = [(sum(len(r["text"]) for r in h.get(k, [])), i, h.get(k, []))
                 for i, h in enumerate(buckets)]
        n, i, rows = max(cands)
        if n == 0:
            continue
        for r in rows:
            picked.append({**r, "src": f"pass{i+1}"})

    picked = merge_rows(picked)

    for item in patch:
        a, b = item["span"]
        if in_any(a, breaks):
            continue
        best, best_n = None, 0
        for att in item["attempts"]:
            cand = [{"start": r["start"], "end": r["end"], "text": norm(r["text"]),
                     "words": r.get("words") or [], "src": "patch"}
                    for r in att["rows"]
                    if a - 0.5 <= r["start"] <= b + 0.5 and usable(norm(r["text"]), r["start"])]
            n = sum(len(r["text"]) for r in cand)
            if n > best_n:
                best, best_n = cand, n
        if not best:
            continue
        have = sum(len(r["text"]) for r in picked if r["start"] < b and r["end"] > a)
        if best_n > have * 1.5 + 10:
            picked = [r for r in picked if not (r["start"] >= a - 0.5 and r["end"] <= b + 0.5)]
            picked.extend(best)

    return merge_rows(picked)


# ---------------------------------------------------------------- 出稿

DEFAULT_LEAD = ["那么", "所以说", "但是", "如果说", "因为", "另外", "接下来",
                "第一个", "第二个", "第三个", "比如说", "也就是说",
                "总之", "最后", "首先", "其次"]


def insert_clause_breaks(t, lead=None, min_run=16, max_run=38):
    """长串里按连接词补逗号。只加标点，绝不删字。"""
    lead = tuple(sorted(lead or DEFAULT_LEAD, key=len, reverse=True))
    out, run, i = [], 0, 0
    while i < len(t):
        ch = t[i]
        if ch in "，。！？、；：":
            out.append(ch); run = 0; i += 1; continue
        if run >= min_run and not (i and t[i - 1] in CLAUSE_GLUE):
            for w in lead:
                if t.startswith(w, i):
                    out.append("，"); run = 0
                    break
        if run >= max_run and ch in "了呢吧啊嘛":
            out.append(ch); out.append("，"); run = 0; i += 1; continue
        out.append(ch); run += 1; i += 1
    return "".join(out)


def gap_thresholds(rows, lo=0.25, hi=0.60):
    """从讲者自己的词间停顿分布里取标点阈值，而不是拍一个固定秒数。

    实测固定 0.35/0.9 秒在慢速歌唱这类慢速音频上会把句号插进词中间
    （「「祈请」被从中间劈开」把「祈请」劈开了）；
    反过来对语速快的讲者，0.35 秒又长到一个逗号都断不出来。
    """
    gaps = []
    for r in rows:
        ws = r.get("words") or []
        gaps.extend(max(0.0, b.get("start", 0.0) - a.get("end", 0.0))
                    for a, b in zip(ws, ws[1:]))
    gaps = sorted(g for g in gaps if g > 0)
    if len(gaps) < 20:
        return 0.35, 0.90
    q = lambda p: gaps[min(len(gaps) - 1, int(len(gaps) * p))]
    comma = max(lo, q(0.80))
    period = max(comma * 1.6, hi, q(0.95))
    return round(comma, 3), round(period, 3)


def punctuate_row(row, comma_gap=0.35, period_gap=0.9):
    """按**词间停顿**在段内补标点。

    过去停顿标点只作用在段与段之间，段内一个字都不加，
    所以 30 秒一整块的段落必然是 90 字裸奔（实测 51 字才有一个标点）。

    词表与正文可能不等长（正文过了繁简转换与术语替换），所以按字数比例映射，
    而不是拿词去重建正文 —— 保证一个字都不会被改动。
    """
    words = row.get("words") or []
    text = row.get("text", "")
    total = sum(len(w.get("word", "")) for w in words)
    if not words or not total or not text:
        return text

    cuts, acc = [], 0
    for prev, cur in zip(words, words[1:]):
        acc += len(prev.get("word", ""))
        gap = cur.get("start", 0.0) - prev.get("end", 0.0)
        mark = "。" if gap >= period_gap else ("，" if gap >= comma_gap else None)
        if mark:
            pos = round(acc * len(text) / total)
            if 0 < pos < len(text):
                cuts.append((pos, mark))

    out = text
    for pos, mark in sorted(cuts, reverse=True):
        if out[pos - 1] in "，。！？、；：" or out[pos] in "，。！？、；：":
            continue
        out = out[:pos] + mark + out[pos:]
    return out


def resplit_rows(rows, max_dur=8.0, max_chars=30):
    """把合并稿重切成互不重叠的字幕条。

    实测旧字幕 19% 与前一条时间重叠、36% 超过 15 秒、最长一条 296 秒——
    根因是两路切分点不同，合并只裁文本、从不重算时间。

    有词级时间戳就按词切，每条的时间是那几个字自己的时间；
    没有就**保持原样**只做防重叠——按比例摊时间是编造，
    对那些吞了静音的长段尤其会把字放到错误的位置上。
    """
    cues = []
    for r in sorted(rows, key=lambda x: (x["start"], x["end"])):
        cues.extend(_split_row(r, max_dur, max_chars))
    cues.sort(key=lambda c: (c["start"], c["end"]))

    out, t = [], 0.0
    for c in cues:
        s = max(c["start"], t)
        e = max(c["end"], s + 0.3)
        out.append({**c, "start": s, "end": e})
        t = e
    return out


def _split_row(r, max_dur, max_chars):
    words = r.get("words") or []
    text = r.get("text", "")
    total = sum(len(w.get("word", "")) for w in words)
    if not words or not total or not text:
        return [dict(r)]

    groups, cur = [], []
    for w in words:
        if cur and (w.get("end", 0.0) - cur[0].get("start", 0.0) > max_dur
                    or sum(len(x.get("word", "")) for x in cur) >= max_chars):
            groups.append(cur); cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    out, acc = [], 0
    for g in groups:
        a = round(acc * len(text) / total)
        acc += sum(len(w.get("word", "")) for w in g)
        b = round(acc * len(text) / total)
        if b <= a:
            continue
        out.append({**r, "start": g[0].get("start", r["start"]),
                    "end": g[-1].get("end", r["end"]), "text": text[a:b]})
    return out or [dict(r)]


def raw_text(passes):
    """各路原始正文（只做繁简转换，不做术语替换）——统计术语命中要用它。"""
    import opencc
    cc = opencc.OpenCC("t2s")
    return "".join(cc.convert(s["text"]) for p in passes for s in p["segments"])


def terms_hits(text, terms):
    """每条术语修正在正文里命中几次。

    README 教人加条目前先手工统计频次——这件事该由代码来做，
    命中 0 次的条目是白写的，命中数异常高的多半是误替换。
    """
    return {a: text.count(a) for a, _ in terms.get("fixes", [])}


def render(rows, dur, outdir, title, terms, meta):
    lead = terms.get("clause_lead") or DEFAULT_LEAD

    def tidy(t):
        t = insert_clause_breaks(t, lead)
        t = re.sub(r"[，、]{2,}", "，", t)
        t = re.sub(r"[，、]+([。！？])", r"\1", t)
        t = re.sub(r"[。]{2,}", "。", t)
        t = re.sub(r"^[，、。]+", "", t)
        t = t.rstrip("，、")
        if t and t[-1] not in "，。！？":
            t += "。"
        return t

    cg, pg = gap_thresholds(rows)
    paras, cur, cur_start, prev_end = [], [], None, None
    for r in rows:
        # 段内先按词间停顿补标点，再交给 tidy 收尾
        t = punctuate_row(r, cg, pg).strip()
        if not t:
            continue
        gap = (r["start"] - prev_end) if prev_end is not None else 0.0
        if cur and (gap >= 1.5 or sum(len(x) for x in cur) >= 220):
            paras.append((cur_start, prev_end, "".join(cur)))
            cur, cur_start = [], None
        if cur_start is None:
            cur_start = r["start"]
        elif gap >= 0.30:
            if not (cur and cur[-1] and cur[-1][-1] in "，。！？、；："):
                cur.append("，" if gap < 0.80 else "。")
        cur.append(t)
        prev_end = r["end"]
    if cur:
        paras.append((cur_start, prev_end, "".join(cur)))
    paras = [(a, b, tidy(t)) for a, b, t in paras]
    nchar = sum(len(t) for _, _, t in paras)

    md = [f"# {title}", "", "**录音全文转录**", "",
          f"> 录音时长 {hms(dur)}（{dur/3600:.2f} 小时）　正文约 {nchar:,} 字　共 {len(paras)} 段  ",
          "> 方括号内为录音时间戳，可据此回听核对原音。", "", "---", ""]
    for a, b, t in paras:
        md.append(f"**[{hms(a)}]**　{t}\n")
    md += ["\n---\n", "## 附：转录说明\n", meta]
    open(os.path.join(outdir, f"{title}_全文转录.md"), "w", encoding="utf-8").write("\n".join(md))

    with open(os.path.join(outdir, f"{title}_全文转录.txt"), "w", encoding="utf-8") as f:
        f.write(f"{title}　录音全文转录\n时长 {hms(dur)}　约 {nchar:,} 字\n\n")
        for a, b, t in paras:
            f.write(f"[{hms(a)}] {t}\n\n")

    def st(x):
        ms = int(round(x * 1000))
        return f"{ms//3600000:02d}:{ms%3600000//60000:02d}:{ms%60000//1000:02d},{ms%1000:03d}"

    cues = resplit_rows(rows)
    with open(os.path.join(outdir, f"{title}_字幕.srt"), "w", encoding="utf-8") as f:
        for i, r in enumerate(cues, 1):
            f.write(f"{i}\n{st(r['start'])} --> {st(r['end'])}\n{r['text']}\n\n")
    return len(paras), nchar, len(cues)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="长音频转录流水线（以不遗漏为目标）")
    ap.add_argument("audio")
    ap.add_argument("-o", "--outdir", default=None, help="输出目录，默认与音频同名")
    ap.add_argument("--profile", default="meeting", choices=list(PROFILES))
    ap.add_argument("--terms", default=None, help="术语修正表 json")
    ap.add_argument("--title", default=None)
    ap.add_argument("--keep-break", action="store_true", help="不剔除中场休息段")
    ap.add_argument("--model", default="large-v3", help="faster-whisper 模型名")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--compute", default=None, help="覆盖档位里的 compute_type")
    ap.add_argument("--language", default="zh")
    args = ap.parse_args()

    ensure_cuda_libs()

    src = os.path.abspath(args.audio)
    title = args.title or os.path.splitext(os.path.basename(src))[0]
    outdir = os.path.abspath(args.outdir or os.path.join(os.path.dirname(src), title + "_转录"))
    work = os.path.join(outdir, ".work")
    os.makedirs(work, exist_ok=True)

    terms = json.load(open(args.terms, encoding="utf-8")) if args.terms else {}
    cfg = PROFILES[args.profile]
    compute = args.compute or (cfg["compute"] if args.device != "cpu" else "int8")
    mk = dict(model_name=args.model, device=args.device, language=args.language)
    log(f"档位 {args.profile}　模型 {args.model}／{args.device}／{compute}　输出 {outdir}")

    wav = prepare_audio(src, work)
    loud = Loudness(wav)

    passes = [transcribe_pass(wav, os.path.join(work, "pass1.json"),
                              cfg["chunk_coarse"], cfg["batch"], cfg["beam"], compute, **mk)]
    if cfg["two_pass"]:
        passes.append(transcribe_pass(wav, os.path.join(work, "pass2.json"),
                                      cfg["chunk_fine"], cfg["batch"], cfg["beam"],
                                      compute, **mk))

    dur = passes[0]["duration"]
    rep = audit(passes[0], loud)
    n_starved = sum(1 for _, _, lab in rep["spans"] if lab == "段内饥饿")
    log(f"审计：覆盖 {rep['cover_pct']:.1f}%　讲话中位音量 {rep['speech_db']:.1f}dB　"
        f"待补 {len(rep['spans'])} 处（其中段内饥饿 {n_starved}）　幻觉 {len(rep['hallu'])} 处")

    breaks = [] if args.keep_break else find_breaks(passes[0], loud)
    for a, b in breaks:
        log(f"识别到休息段 {hms(a)}–{hms(b)}（{(b-a)/60:.1f} 分钟），将剔除")
    if breaks:
        rep["spans"] = [s for s in rep["spans"] if not in_any(s[0], breaks)]

    patch = repatch(wav, rep["spans"], os.path.join(work, "repatch.json"),
                    compute=compute, **mk)
    rows = combine(passes, patch, terms, breaks, dur, rep.get("drop", []))
    per_pass = ", ".join(
        format(sum(len(s["text"]) for s in p["segments"]), ",") for p in passes)
    log(f"合并后 {len(rows)} 段 / {sum(len(r['text']) for r in rows):,} 字"
        f"（各单路：{per_pass}）")

    # 终审：审的必须是交付物本身。过去这里报的是 pass1 的数（97.3%），
    # 而成品字幕的时间并集只有 90.0%，给读者看的数字对不上交付物。
    fin = audit_rows(rows, dur)
    log(f"终审：合并稿覆盖 {fin['cover_pct']:.1f}%　有效语音 {fin['speech_pct']:.1f}%　"
        f"残余饥饿段 {fin['starved']} 处")
    if fin["starved"]:
        log("  ⚠ 仍有段落时长撑不起字数，补转没能捞回来，出稿前请对照 .srt 回听这些位置")

    # 必须在替换**之前**数：正文里源词早被换掉了，事后统计只会全是 0
    hits = terms_hits(raw_text(passes), terms)
    if hits:
        miss = [k for k, v in hits.items() if v == 0]
        log(f"术语表：{len(hits)} 条，命中 {sum(1 for v in hits.values() if v)} 条"
            + (f"，0 命中 {len(miss)} 条（{'、'.join(miss[:5])}{'…' if len(miss)>5 else ''}）"
               if miss else ""))

    meta_lines = [
        f"**转录方式**　faster-whisper {args.model}，{len(passes)} 路交叉 + "
        f"{len(rep['spans'])} 处定点补转，取并集。\n",
        f"**覆盖率**　本文档时间覆盖 {fin['cover_pct']:.1f}%，其中字数撑得起的有效语音约 "
        f"{fin['speech_pct']:.1f}%（按 {SPEECH_RATE:.0f} 字/秒估）；"
        f"单路原始审计 {rep['cover_pct']:.1f}%，讲话段中位音量 {rep['speech_db']:.1f} dBFS。\n",
    ]
    for a, b in breaks:
        meta_lines.append(f"**已剔除**　休息段 {hms(a)}–{hms(b)}"
                          f"（{(b-a)/60:.1f} 分钟，音量显著低于讲授段，转录输出为幻觉）。\n")
    if rep["hallu"]:
        meta_lines.append(f"**已剔除幻觉片段** {len(rep['hallu'])} 处（静音段被转成网络字幕套语）。\n")
    if fin["starved"]:
        meta_lines.append(f"**待人工核对** {fin['starved']} 处段落时长与字数明显不匹配，"
                          "疑似仍有未转出的内容，请对照字幕回听。\n")
    meta_lines.append("**注意**　语音识别对专有名词与专业术语存在同音误识，"
                      "已按术语表统一校正可确定者，不能确定者保留原样。"
                      "段内标点由词间停顿与连接词推定，仅供阅读参考，不代表讲者原意停顿。\n")

    n_para, nchar, n_cue = render(rows, dur, outdir, title, terms, "\n".join(meta_lines))

    json.dump({"audit": {k: v for k, v in rep.items() if k != "hallu"},
               "final": fin, "terms_hits": hits,
               "hallucinations": rep["hallu"], "breaks": breaks},
              open(os.path.join(outdir, "质检报告.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    loud.close()
    log(f"完成：{n_para} 段 / {nchar:,} 字 / {n_cue} 条字幕")
    log(f"  {outdir}/{title}_全文转录.md")
    log(f"  {outdir}/{title}_全文转录.txt")
    log(f"  {outdir}/{title}_字幕.srt")
    log(f"  {outdir}/质检报告.json")


if __name__ == "__main__":
    main()
