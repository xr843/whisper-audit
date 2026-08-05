"""多路转录结果合并。"""
from .audit import HALLU_PAT, in_any


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
