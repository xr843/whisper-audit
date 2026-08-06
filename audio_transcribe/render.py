"""标点、切字幕、出稿。"""
import os
import re

from . import hms

# 连接词前不能断句的粘连字。实测 15 处「这是，第一个环节」都是因为
# 只看连接词、不看前一个字——第一个/第二个在汉语里更多是宾语而非句首。
#
# 只收**前置**成分（是第一个／就比如说／是因为）。
# 「的」「了」是句末助词，方向正好相反——它们后面恰恰该断句
# （的，所以说／了，所以说／的，另外），把它们收进来会挡掉 16 个正确的逗号。
CLAUSE_GLUE = "是为把被将对与和及或即含有到就都也很才又更最称叫做算"


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
    （实测把「祈请」这个双字词从中间劈开，前字后面被插了句号）；
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
