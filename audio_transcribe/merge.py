"""多路转录结果合并。"""
from .audit import HALLU_PAT, in_any
from .terms import pinyin_fix


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


def combine(passes, patch, terms, breaks, dur, drop_spans=(),
            pinyin=False, loose=False, return_hits=False):
    """逐 30 秒窗口在各路之间取字数多的，再用补转填两路都空的洞。

    `pinyin=True` 时在字面替换之后再跑一遍拼音模糊纠错。
    `loose` 默认 **False**：近音归并（zh/z、ang/an…）碰撞面大得多，
    实测能多修几处，但也更容易在词边界上误伤，要用得显式开。

    `return_hits=True` 时一并返回拼音纠错的改动清单，写进质检报告——
    凡是自动改正文的逻辑都必须留痕。
    """
    import opencc
    cc = opencc.OpenCC("t2s")
    fixes = terms.get("fixes", [])

    def norm(t):
        """返回 (规范化后的文本, 拼音纠错改动清单)。

        改动清单按行携带，最后只从**进入成品的行**收集——
        补转会为同一区段生成多组候选，落选的那些也会走 norm，
        若在这里直接累加，质检报告会列出根本没进成品的修改。
        """
        t = cc.convert(t).strip()
        for a, b in fixes:          # 字面替换优先：人工逐条确认过的
            t = t.replace(a, b)
        if not pinyin:
            return t, []
        return pinyin_fix(t, terms, loose=loose)

    def usable(t, start):
        if not t or HALLU_PAT.search(t):
            return False
        if in_any(start, breaks):
            return False
        # 落在休息段之外的零散幻觉（音量低+低置信），也要剔除
        if in_any(start, drop_spans, pad=0.2):
            return False
        return True

    # norm() 每条只调一次——它带副作用（累积拼音纠错的改动清单），
    # 调两次会让记账翻倍。
    prepped = []
    for p in passes:
        rows = []
        for s in p["segments"]:
            t, h = norm(s["text"])
            if usable(t, s["start"]):
                rows.append({"start": s["start"], "end": s["end"], "text": t,
                             "words": s.get("words") or [], "_pyhits": h})
        prepped.append(rows)

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
            cand = []
            for r in att["rows"]:
                if not (a - 0.5 <= r["start"] <= b + 0.5):
                    continue
                t, h = norm(r["text"])
                if usable(t, r["start"]):
                    cand.append({"start": r["start"], "end": r["end"], "text": t,
                                 "words": r.get("words") or [], "src": "patch",
                                 "_pyhits": h})
            n = sum(len(r["text"]) for r in cand)
            if n > best_n:
                best, best_n = cand, n
        if not best:
            continue
        have = sum(len(r["text"]) for r in picked if r["start"] < b and r["end"] > a)
        if best_n > have * 1.5 + 10:
            picked = [r for r in picked if not (r["start"] >= a - 0.5 and r["end"] <= b + 0.5)]
            picked.extend(best)

    rows = merge_rows(picked)

    # 记账要按**最终文本**复核。行进了成品，不等于那处改动还在行里：
    # merge_rows 会用 strip_common 裁掉重复字块，被纠错的词可能正好在裁掉的那段。
    # 实测两路高度重叠时，1 处真实改动会记出 2 条，第二条指向一个
    # 压根没有该词的行——审计人照着时间戳去回听，什么也找不到。
    #
    # 同时丢掉 pos：它是裁剪前的行内偏移，裁完就失效了，留着比没有更误导。
    # 改带所在行的起始时间——「可追溯」要能落到音频上的一个时刻才算数。
    hits = []
    for r in rows:
        left = {}
        for h in r.get("_pyhits", []):
            cap = r["text"].count(h["to"])
            used = left.get(h["to"], 0)
            if used < cap:
                left[h["to"]] = used + 1
                hits.append({"from": h["from"], "to": h["to"],
                             "t": round(r["start"], 2)})
        r.pop("_pyhits", None)
    return (rows, hits) if return_hits else rows
