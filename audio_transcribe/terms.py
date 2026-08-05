"""术语表：字面精确替换 + 拼音模糊匹配。

字面替换（fixes）穷举不完误识写法——实测 54 条里 12 条从未命中。
拼音匹配只需列正确写法，一条覆盖一整类同音错。

护栏：只在拼音一致（或近音，可关）时替换，所有改动记账。
拼音不同的词一律不许碰——这是这个项目栽过两次的坑（静悄悄改内容）。
"""
from .evaluate import pinyin_key


def pinyin_fix(text, terms, loose=False):
    words = [w for w in (terms.get("terms") or []) if len(w) >= 2]
    if not words or not text:
        return text, []
    index = {}
    for w in words:
        index.setdefault((len(w), pinyin_key(w, loose)), w)
    lengths = sorted({len(w) for w in words}, reverse=True)   # 长词优先

    out, hits, i = [], [], 0
    while i < len(text):
        for L in lengths:
            seg = text[i:i + L]
            if len(seg) < L:
                continue
            w = index.get((L, pinyin_key(seg, loose)))
            if w is None:
                continue
            if seg != w:
                hits.append({"pos": i, "from": seg, "to": w})
            out.append(w)
            i += L
            break
        else:
            out.append(text[i])
            i += 1
    return "".join(out), hits
