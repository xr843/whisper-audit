"""术语表：字面精确替换 + 拼音模糊匹配。

字面替换（fixes）穷举不完误识写法——实测 51 条里 16 条从未命中。
拼音匹配只需列正确写法，一条覆盖一整类同音错。

护栏，从外到内三层：

1. 只在拼音一致（或近音，显式开）时替换，拼音不同的词一律不许碰
2. **词边界保护**（2026-08-06 加）：替换区间必须与 jieba 分词边界对齐——
   起点是某个词的词首、终点是某个词的词尾。没有这层时，扫描器会在
   「活动场所|的|税务」的接缝上凑出「所的税」替换成「所得税」，把正确
   文本改坏；穷举扫描发现这类跨词凑字组合有 1000+。边界保护把这一整类堵住。
3. 所有改动记账（位置、原文、替换后），写进质检报告可逐条回听

**边界保护堵不住「整词同音」**：「节余分配→结余分配」里节余/分配各自是
完整词，边界完全合法，但节余/结余是两个真实概念——这是无调拼音粒度的
极限，也是 pinyin_fix 默认关闭的原因（tests 里有 xfail 盯着这条）。
"""
from .evaluate import pinyin_key

_tokenizer = None


def _word_bounds(text):
    """返回 (词首位置集合, 词尾位置集合)。jieba 延迟加载（首次约 0.3 秒）。"""
    global _tokenizer
    if _tokenizer is None:
        import jieba
        _tokenizer = jieba
    starts, ends, p = set(), set(), 0
    for w in _tokenizer.cut(text):
        starts.add(p)
        p += len(w)
        ends.add(p)
    return starts, ends


def pinyin_fix(text, terms, loose=False, boundary=True):
    """`boundary=False` 关掉词边界保护——只应在对照实验里用，生产不要关。"""
    words = [w for w in (terms.get("terms") or []) if len(w) >= 2]
    if not words or not text:
        return text, []
    index = {}
    for w in words:
        index.setdefault((len(w), pinyin_key(w, loose)), w)
    lengths = sorted({len(w) for w in words}, reverse=True)   # 长词优先

    starts, ends = _word_bounds(text) if boundary else (None, None)

    out, hits, i = [], [], 0
    while i < len(text):
        for L in lengths:
            seg = text[i:i + L]
            if len(seg) < L:
                continue
            w = index.get((L, pinyin_key(seg, loose)))
            if w is None:
                continue
            if boundary and seg != w and not (i in starts and i + L in ends):
                continue      # 候选横跨词边界：是接缝上凑出来的字串，不是词
            if seg != w:
                hits.append({"pos": i, "from": seg, "to": w})
            out.append(w)
            i += L
            break
        else:
            out.append(text[i])
            i += 1
    return "".join(out), hits
