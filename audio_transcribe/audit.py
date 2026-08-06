"""覆盖率审计：找空洞、量音量、揪幻觉。"""
import math
import re

# 模型在静音段会吐训练数据里的网络视频字幕套语。这些词出现即幻觉。
# 词表来源：Whisper 中文训练数据里的高频视频字幕套语（实测静音段幻觉）。
HALLU_WORDS = (r"点赞|订阅|打赏|字幕志愿者|中文字幕|谢谢观看|感谢观看|"
               r"请不吝|字幕组|翻译by|本视频|下期再见")
HALLU_PAT = re.compile(HALLU_WORDS)

# 正常汉语讲授的语速。实测这份 3.65 小时讲座的 segment 中位密度是 3.63 字/秒，
# 取 3.0 作保守基准：用来估「有效语音时长」和判定段内饥饿。
SPEECH_RATE = 3.0


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
