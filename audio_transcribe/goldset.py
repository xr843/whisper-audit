"""金标生成与评测入口。

金标格式是两列 TSV：`秒数<TAB>文本`。初稿直接取 ASR 结果，
**人工只改错字**——不碰格式、不打时间戳、不做对齐。

这是整个正确率体系里唯一需要人力的环节，所以要把它压到最省力。
"""
import re

from .evaluate import score

_CUE = re.compile(r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)\s*-->\s*"
                  r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)")


def _sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_hms(text):
    """接受 00:12:34、12:34、754 三种写法，返回秒数。None/空串返回 None。"""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return sec


def parse_srt(path):
    rows = []
    for block in open(path, encoding="utf-8").read().strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        m = _CUE.search(lines[1])
        if not m:
            continue
        g = m.groups()
        rows.append({"start": _sec(*g[:4]), "end": _sec(*g[4:]),
                     "text": "".join(lines[2:]).strip()})
    return rows


def make_goldset(srt_path, start=None, end=None):
    out = []
    for r in parse_srt(srt_path):
        if start is not None and r["start"] < start:
            continue
        if end is not None and r["start"] >= end:
            continue
        out.append((r["start"], r["end"], r["text"]))
    return out


def write_tsv(rows, path):
    """三列：起始秒 <TAB> 结束秒 <TAB> 文本。人只该动第三列。"""
    with open(path, "w", encoding="utf-8") as f:
        for a, b, text in rows:
            f.write(f"{a}\t{b}\t{text}\n")


def read_tsv(path):
    """容忍人工编辑留下的多余空格与空行——人只该操心错字。

    两列的旧格式（起始秒 + 文本）也读得进来，结束时间退化为起始时间。
    """
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            rows.append((float(parts[0].strip()), float(parts[1].strip()),
                         "\t".join(parts[2:]).strip()))
        else:
            a = float(parts[0].strip())
            rows.append((a, a, parts[1].strip() if len(parts) > 1 else ""))
    return rows


def evaluate_srt(gold_path, hyp_srt_path):
    """金标 vs 成品字幕。

    窗口取金标的 [最早 start, 最晚 end]，hyp 按**时间重叠**选取而不是
    「start 落在窗口内」。差别在跨版本对比时是致命的：

        金标来自 v1，末条 "……内容是所得税优惠政策" @6.5s
        v2 把同一句切成两条：@6.5s "……内容是" / @8.2s "所得税优惠政策"

    按 start 落窗口内选取，第二条（8.2s > 金标末条 start 6.5s）会被整条丢掉，
    于是一字不差的 v2 被判出 7 个删除错、CER 35%。而删除率正是这个项目的
    招牌指标，「拿金标评测后续版本」又正是整套金标体系存在的理由。

    比较的是**拼接后的整段文本**，不做逐条对齐——逐条对齐会把切分差异
    算成错误，而切分本来就允许不同。
    """
    gold = read_tsv(gold_path)
    if not gold:
        raise ValueError(f"金标为空：{gold_path}")
    lo = min(a for a, _, _ in gold)
    hi = max(b for _, b, _ in gold)
    ref = "".join(text for _, _, text in gold)
    hyp = "".join(r["text"] for r in parse_srt(hyp_srt_path)
                  if r["end"] > lo and r["start"] < hi)
    rep = score(ref, hyp)
    rep["window"] = [lo, hi]
    return rep


def find_srt(outdir):
    """输出目录里找唯一的 .srt。"""
    import glob
    import os
    hits = sorted(glob.glob(os.path.join(outdir, "*.srt")))
    if not hits:
        raise FileNotFoundError(f"{outdir} 里没有 .srt")
    if len(hits) > 1:
        raise ValueError(f"{outdir} 里有多个 .srt，请用 --hyp 指定具体文件：{hits}")
    return hits[0]


def format_report(rep):
    n = rep["n_ref"]
    lines = [
        f"参考 {n:,} 字　假设 {rep['n_hyp']:,} 字",
        "",
        f"  CER          {rep['cer']*100:6.2f}%   ← 总错误率",
        f"  替换 sub     {rep['sub']:6,}   {100*rep['sub']/n if n else 0:5.2f}%",
        f"  删除 dele    {rep['dele']:6,}   {100*rep['dele']/n if n else 0:5.2f}%"
        "   ← 「不遗漏」的直接度量",
        f"  插入 ins     {rep['ins']:6,}   {100*rep['ins']/n if n else 0:5.2f}%",
        "",
        f"  替换错误里同音 {rep['homo']:,} 处（{rep['homo_pct']:.1f}%）、"
        f"近音 {rep['near']:,} 处（{rep['near_pct']:.1f}%）",
        "  ↑ 这两个数就是拼音纠错与 LLM 同音校订的天花板",
    ]
    return "\n".join(lines)
