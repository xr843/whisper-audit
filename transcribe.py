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
import argparse, json, os, re, subprocess, sys, time, wave


def ensure_cuda_libs():
    """CTranslate2 要在运行时找到 cuBLAS/cuDNN，它们随 torch 的 pip 包装在 nvidia/ 下。
    LD_LIBRARY_PATH 必须在进程启动前生效，所以这里设好后重启自己。"""
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


ensure_cuda_libs()

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
        import math
        return 20 * math.log10(max(float((a ** 2).mean()) ** 0.5, 1e-9) / 32768)


# ---------------------------------------------------------------- 转录

def transcribe_pass(wav, out_json, chunk_length, batch, beam, compute="float16"):
    """一路转录。

    刻意不传 initial_prompt：实测模型会在静音段把 prompt 原样"转录"出来，
    还占掉真实语音的时间段（有个案例吞了 88 秒）。术语引导交给事后的术语表。"""
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
            model = WhisperModel("large-v3", device="cuda", compute_type=compute)
            pipe = BatchedInferencePipeline(model=model)
            t0 = time.time()
            segs, info = pipe.transcribe(
                wav, language="zh", batch_size=attempt_batch, beam_size=beam,
                vad_filter=True,
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
                             "no_speech_prob": round(s.no_speech_prob, 4)})
                if s.end - last >= 600:
                    last = s.end
                    el = time.time() - t0
                    log(f"  {100*s.end/info.duration:5.1f}%  {s.end/max(el,1e-9):.1f}x")
            el = time.time() - t0
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or attempt_batch == max(1, batch // 4):
                raise
            log(f"  显存不足，降到 batch={attempt_batch//2} 重试")
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

    spans.sort()
    return {"duration": dur, "cover": cover, "cover_pct": 100 * cover / dur,
            "speech_db": speech_db, "gaps": gaps, "spans": spans, "drop": drop,
            "hallu": [(s["start"], s["end"], s["text"], db) for s, db in hallu]}


def find_break(data, loud, min_len=120, step=10):
    """找中场休息。

    判据必须是「音量低」**且**「转出的是幻觉或极少字」——只看音量会两头判偏：
    实测有一版把休息段末尾多划了 47 秒，把"接下来我们开始互动的环节"整句删掉了。
    步长取 10 秒而非 30 秒，边界才够准。
    """
    dur = data["duration"]
    segs = data["segments"]

    dbs = [(t, loud.db(t, t + step)) for t in range(0, int(dur), step)]
    vals = sorted(d for _, d in dbs if d == d)
    if not vals:
        return None
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

    best, cur = None, None
    for i, f in enumerate(flags):
        if f:
            cur = (dbs[i][0], dbs[i][0] + step) if cur is None else (cur[0], dbs[i][0] + step)
            if best is None or (cur[1] - cur[0]) > (best[1] - best[0]):
                best = cur
        else:
            cur = None
    if best is None or (best[1] - best[0]) < min_len:
        return None
    return best


# ---------------------------------------------------------------- 补转

def repatch(wav, spans, out_json, pad=6.0):
    """只对问题区段重转。关 VAD（它已经判错一次了）、抬高 no_speech 门槛、给足上下文。"""
    if not spans:
        return []
    if os.path.exists(out_json):
        log(f"复用已有 {os.path.basename(out_json)}")
        return json.load(open(out_json, encoding="utf-8"))
    from faster_whisper import WhisperModel
    log(f"定点补转 {len(spans)} 处（{sum(b-a for a,b,_ in spans):.0f} 秒）…")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    res = []
    for i, (a, b, label) in enumerate(spans):
        a2, b2 = max(0, a - pad), b + pad
        tmp = f"/tmp/_span_{os.getpid()}_{i:03d}.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", wav,
                        "-ss", str(a2), "-to", str(b2), "-c", "copy", tmp], check=True)
        attempts = []
        for mode, kw in (("novad", dict(vad_filter=False)),
                         ("vad_loose", dict(vad_filter=True,
                                            vad_parameters=dict(min_silence_duration_ms=2000,
                                                                threshold=0.2, speech_pad_ms=800)))):
            segs, _ = model.transcribe(tmp, language="zh", beam_size=5,
                                       condition_on_previous_text=False,
                                       temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                       no_speech_threshold=0.9,   # 宁可多转，不可漏转
                                       log_prob_threshold=-2.0, **kw)
            attempts.append({"mode": mode, "rows": [
                {"start": round(a2 + s.start, 2), "end": round(a2 + s.end, 2),
                 "text": s.text.strip()} for s in segs]})
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
                if len(strip_common(prev["text"], cur, 10)) < len(cur) * 0.3:
                    cur = ""
            if len(cur.strip()) < 3:
                break
        cur = cur.strip()
        if len(cur) >= 3:
            out.append({**r, "text": cur})
    return out


def combine(passes, patch, terms, break_span, dur, drop_spans=()):
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
        if break_span and break_span[0] <= start <= break_span[1]:
            return False
        # 落在休息段之外的零散幻觉（音量低+低置信），也要剔除
        for a, b in drop_spans:
            if a - 0.2 <= start <= b + 0.2:
                return False
        return True

    prepped = []
    for p in passes:
        prepped.append([{"start": s["start"], "end": s["end"], "text": norm(s["text"])}
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
        if break_span and break_span[0] <= a <= break_span[1]:
            continue
        best, best_n = None, 0
        for att in item["attempts"]:
            cand = [{"start": r["start"], "end": r["end"], "text": norm(r["text"]), "src": "patch"}
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

def render(rows, dur, outdir, title, terms, meta):
    lead = tuple(sorted(terms.get("clause_lead",
                 ["那么", "所以说", "但是", "如果说", "因为", "另外", "接下来",
                  "第一个", "第二个", "第三个", "比如说", "也就是说",
                  "总之", "最后", "首先", "其次"]), key=len, reverse=True))

    def clauses(t, min_run=16, max_run=38):
        out, run, i = [], 0, 0
        while i < len(t):
            ch = t[i]
            if ch in "，。！？、；：":
                out.append(ch); run = 0; i += 1; continue
            if run >= min_run:
                for w in lead:
                    if t.startswith(w, i):
                        out.append("，"); run = 0
                        break
            if run >= max_run and ch in "了呢吧啊嘛":
                out.append(ch); out.append("，"); run = 0; i += 1; continue
            out.append(ch); run += 1; i += 1
        return "".join(out)

    def tidy(t):
        t = clauses(t)
        t = re.sub(r"[，、]{2,}", "，", t)
        t = re.sub(r"[，、]+([。！？])", r"\1", t)
        t = re.sub(r"[。]{2,}", "。", t)
        t = re.sub(r"^[，、。]+", "", t)
        t = t.rstrip("，、")
        if t and t[-1] not in "，。！？":
            t += "。"
        return t

    paras, cur, cur_start, prev_end = [], [], None, None
    for r in rows:
        t = r["text"].strip()
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

    with open(os.path.join(outdir, f"{title}_字幕.srt"), "w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            f.write(f"{i}\n{st(r['start'])} --> {st(r['end'])}\n{r['text']}\n\n")
    return len(paras), nchar


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="长音频转录流水线（以不遗漏为目标）")
    ap.add_argument("audio")
    ap.add_argument("-o", "--outdir", default=None, help="输出目录，默认与音频同名")
    ap.add_argument("--profile", default="meeting", choices=list(PROFILES))
    ap.add_argument("--terms", default=None, help="术语修正表 json")
    ap.add_argument("--title", default=None)
    ap.add_argument("--keep-break", action="store_true", help="不剔除中场休息段")
    args = ap.parse_args()

    src = os.path.abspath(args.audio)
    title = args.title or os.path.splitext(os.path.basename(src))[0]
    outdir = os.path.abspath(args.outdir or os.path.join(os.path.dirname(src), title + "_转录"))
    work = os.path.join(outdir, ".work")
    os.makedirs(work, exist_ok=True)

    terms = json.load(open(args.terms, encoding="utf-8")) if args.terms else {}
    cfg = PROFILES[args.profile]
    log(f"档位 {args.profile}　输出 {outdir}")

    wav = prepare_audio(src, work)
    loud = Loudness(wav)

    passes = [transcribe_pass(wav, os.path.join(work, "pass1.json"),
                              cfg["chunk_coarse"], cfg["batch"], cfg["beam"], cfg["compute"])]
    if cfg["two_pass"]:
        passes.append(transcribe_pass(wav, os.path.join(work, "pass2.json"),
                                      cfg["chunk_fine"], cfg["batch"], cfg["beam"],
                                      cfg["compute"]))

    dur = passes[0]["duration"]
    rep = audit(passes[0], loud)
    log(f"审计：覆盖 {rep['cover_pct']:.1f}%　讲话中位音量 {rep['speech_db']:.1f}dB　"
        f"待补 {len(rep['spans'])} 处　幻觉 {len(rep['hallu'])} 处")

    brk = None if args.keep_break else find_break(passes[0], loud)
    if brk:
        log(f"识别到中场休息 {hms(brk[0])}–{hms(brk[1])}（{(brk[1]-brk[0])/60:.1f} 分钟），将剔除")
        rep["spans"] = [s for s in rep["spans"] if not (brk[0] <= s[0] <= brk[1])]

    patch = repatch(wav, rep["spans"], os.path.join(work, "repatch.json"))
    rows = combine(passes, patch, terms, brk, dur, rep.get("drop", []))
    per_pass = ", ".join(
        format(sum(len(s["text"]) for s in p["segments"]), ",") for p in passes)
    log(f"合并后 {len(rows)} 段 / {sum(len(r['text']) for r in rows):,} 字"
        f"（各单路：{per_pass}）")

    meta_lines = [
        f"**转录方式**　faster-whisper large-v3，{len(passes)} 路交叉 + {len(rep['spans'])} 处定点补转，取并集。\n",
        f"**覆盖率**　单路审计 {rep['cover_pct']:.1f}%，讲话段中位音量 {rep['speech_db']:.1f} dBFS。\n",
    ]
    if brk:
        meta_lines.append(f"**已剔除**　中场休息 {hms(brk[0])}–{hms(brk[1])}"
                          f"（{(brk[1]-brk[0])/60:.1f} 分钟，音量显著低于讲授段，转录输出为幻觉）。\n")
    if rep["hallu"]:
        meta_lines.append(f"**已剔除幻觉片段** {len(rep['hallu'])} 处（静音段被转成网络字幕套语）。\n")
    meta_lines.append("**注意**　语音识别对专有名词与专业术语存在同音误识，"
                      "已按术语表统一校正可确定者，不能确定者保留原样。"
                      "段内标点由停顿与连接词推定，仅供阅读参考，不代表讲者原意停顿。\n")

    n_para, nchar = render(rows, dur, outdir, title, terms, "\n".join(meta_lines))

    json.dump({"audit": {k: v for k, v in rep.items() if k != "hallu"},
               "hallucinations": rep["hallu"], "break": brk},
              open(os.path.join(outdir, "质检报告.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    log(f"完成：{n_para} 段 / {nchar:,} 字")
    log(f"  {outdir}/{title}_全文转录.md")
    log(f"  {outdir}/{title}_全文转录.txt")
    log(f"  {outdir}/{title}_字幕.srt")
    log(f"  {outdir}/质检报告.json")


if __name__ == "__main__":
    main()
