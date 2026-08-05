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
        out.append((r["start"], r["text"]))
    return out


def write_tsv(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for t, text in rows:
            f.write(f"{t}\t{text}\n")


def read_tsv(path):
    """容忍人工编辑留下的多余空格与空行——人只该操心错字。"""
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        t, _, text = line.partition("\t")
        rows.append((float(t.strip()), text.strip()))
    return rows


def evaluate_srt(gold_path, hyp_srt_path):
    """金标 vs 成品字幕。

    时间窗取金标首尾两条的 start，上界用 `<=` 把末条也含进来。
    比较的是**拼接后的整段文本**，不做逐条对齐——逐条对齐会把
    切分差异算成错误，而切分本来就允许不同。
    """
    gold = read_tsv(gold_path)
    if not gold:
        raise ValueError(f"金标为空：{gold_path}")
    lo = min(t for t, _ in gold)
    hi = max(t for t, _ in gold)
    ref = "".join(text for _, text in gold)
    hyp = "".join(r["text"] for r in parse_srt(hyp_srt_path)
                  if lo <= r["start"] <= hi)
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
