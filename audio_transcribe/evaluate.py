"""字级 CER 评测。

流水线过去所有质量话术都是覆盖率——转到了多少；没有任何数字回答转对了多少。
覆盖率 100% 的稿子可以句句是错的。

除了 CER，另报两个本项目专属的数：
  删除率      ——「不遗漏」这个卖点的直接度量，过去只有覆盖率在代理它
  同音/近音率 —— 替换错误里拼音相同或相近的占比，直接给出拼音纠错与
                 LLM 同音校订的天花板，是后续所有取舍的依据
"""
import unicodedata

_PUNCT = set("，。！？、；：“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]{}-~")
_DIGITS = str.maketrans("0123456789", "〇一二三四五六七八九")

_INITIALS = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("n", "l")]
_FINALS = [("ang", "an"), ("eng", "en"), ("ing", "in")]

_cc = None


def normalize(text):
    """繁→简、全角→半角、数字统一、去标点空白。"""
    global _cc
    if _cc is None:
        import opencc
        _cc = opencc.OpenCC("t2s")
    t = unicodedata.normalize("NFKC", _cc.convert(text))
    t = t.translate(_DIGITS)
    return "".join(c for c in t if c not in _PUNCT and not c.isspace())


def pinyin_key(text, loose=False):
    from pypinyin import Style, lazy_pinyin
    syls = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
    if not loose:
        return tuple(syls)
    out = []
    for s in syls:
        for a, b in _INITIALS:
            if s.startswith(a):
                s = b + s[len(a):]
                break
        for a, b in _FINALS:
            if s.endswith(a):
                s = s[:-len(a)] + b
                break
        out.append(s)
    return tuple(out)


def score(ref, hyp):
    from rapidfuzz.distance import Levenshtein
    ref, hyp = normalize(ref), normalize(hyp)
    ops = Levenshtein.editops(ref, hyp)
    sub = dele = ins = homo = near = 0
    for o in ops:
        if o.tag == "delete":
            dele += 1
        elif o.tag == "insert":
            ins += 1
        else:
            sub += 1
            a, b = ref[o.src_pos], hyp[o.dest_pos]
            if pinyin_key(a) == pinyin_key(b):
                homo += 1
                near += 1
            elif pinyin_key(a, loose=True) == pinyin_key(b, loose=True):
                near += 1
    n = len(ref)
    return {"n_ref": n, "n_hyp": len(hyp), "sub": sub, "dele": dele, "ins": ins,
            "cer": (sub + dele + ins) / n if n else 0.0,
            "homo": homo, "near": near,
            "homo_pct": 100 * homo / sub if sub else 0.0,
            "near_pct": 100 * near / sub if sub else 0.0}
