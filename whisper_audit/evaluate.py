"""字级 CER 评测。

流水线过去所有质量话术都是覆盖率——转到了多少；没有任何数字回答转对了多少。
覆盖率 100% 的稿子可以句句是错的。

除了 CER，另报两个本项目专属的数：
  删除率      ——「不遗漏」这个卖点的直接度量，过去只有覆盖率在代理它
  同音/近音率 —— 替换错误里拼音相同或相近的占比，直接给出拼音纠错与
                 LLM 同音校订的天花板，是后续所有取舍的依据
"""
import re
import unicodedata

_PUNCT = set("，。！？、；：“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]{}-~")

# 括号内的纯外文注释：书面文本约定，说话人不会读出来。
# 实测 FLEURS 中文测试集 945 条里 129 条含这类注释（共 1,770 字符，
# 占参考总字数 5%），典型如：
#     参考  特朗普与土耳其总统雷杰普·塔伊普·埃尔多安（Recep Tayyip Erdoğan）通话后…
#     音频  只念了中文译名，Latin 部分根本没说
# 不剔除的话，任何 ASR 都会为这段凭空吃满删除错——whisper large-v3 的
# FLEURS CER 因此从 3.77% 被抬到 7.56%（近两倍），且各引擎受损程度不同，
# 对比会失真。这是参考文本的书面约定，不是模型的错。
_FOREIGN_GLOSS = re.compile(r"[（(]\s*[^（()）]*?[A-Za-zÀ-ÿĀ-ſ][^（()）]*?\s*[)）]")
# 0 映射为「零」而非「〇」，且既有的「〇」也归一为「零」——
# ASR 引擎输出汉字数字时几乎都写「零」，映射到「〇」会让
# 「二零一九」对「2019」平白多出一个替换错，系统性偏袒
# 输出阿拉伯数字的引擎（实测在 FLEURS 双引擎对比中撞出）。
_DIGITS = str.maketrans("0123456789〇", "零一二三四五六七八九零")

_INITIALS = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("n", "l")]
_FINALS = [("ang", "an"), ("eng", "en"), ("ing", "in")]

_cc = None


def normalize(text):
    """繁→简、全角→半角、数字统一、去标点空白。

    数字统一是逐字符映射（str.translate），不是数值转换："25" 会变成
    "二五" 而不是"二十五"。财税类语料里法规条款号、金额、税率常见，
    同一个数用阿拉伯数字和中文数字两种记法写会被记成替换/增删错误——
    这类噪声分不清是转录真错还是记法差异，读 cer/sub/dele 时需留意。
    """
    global _cc
    if _cc is None:
        import opencc
        _cc = opencc.OpenCC("t2s")
    t = unicodedata.normalize("NFKC", _cc.convert(text))
    t = t.translate(_DIGITS)
    return "".join(c for c in t if c not in _PUNCT and not c.isspace())


def strip_foreign_gloss(text):
    """剔除括号内的纯外文注释（含至少一个拉丁字母的括号段）。

    **只该用在参考文本上，且只在公开基准这种「参考取自书面文本」的场景**。
    自建金标是从 ASR 输出改错字来的，不会有这类注释，用不上也不该用。

    含中文的括号（如「（简称甲方）」）不动——那是说话人可能真读出来的。
    """
    return _FOREIGN_GLOSS.sub("", text)


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
    """按字对齐 ref/hyp，返回一条结果的 n_ref/n_hyp/sub/dele/ins/cer/homo/near 等。

    聚合多条结果时：必须先累加各条的 sub/dele/ins/n_ref，再用累加值重算
    cer = (sub+dele+ins)/n_ref，不能对各条的 cer 字段取平均或加权平均
    ——n_ref 为 0（参考文本为空、假设全是幻觉）时本函数按约定返回
    cer=0.0（不是 nan），这条会把平均值直接拉偏。ins 字段不受此影响，
    仍是可靠的幻觉计数。

    sub/dele/ins 的切分基于 Levenshtein 最短编辑路径，同一对文本可能有
    多条编辑距离相同的路径，算法只会返回其中一条，分类因此存在固有歧义。
    典型例子是换位：'猫乙'→'乙猫' 两个字都还在输出里，却会被记成删一个
    插一个（dele=1, ins=1, sub=0）。cer 总量不受影响，但本项目把删除率
    单列当作"不遗漏"的直接度量，换位场景会让这个读数偏紧，不代表真实遗漏。
    """
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
            # 守卫：lazy_pinyin(errors="ignore") 对转不出拼音的字符（拉丁字母、
            # 数字、漏网标点）直接丢弃而非保留，两个这样的字符都会得到空元组
            # ()，而 () == () 为真——不加守卫会把互不相关的非中文替换误判成
            # 同音/近音。只有两侧都真正转出了拼音才允许判等。
            ka, kb = pinyin_key(a), pinyin_key(b)
            if ka and kb and ka == kb:
                homo += 1
                near += 1
            else:
                la, lb = pinyin_key(a, loose=True), pinyin_key(b, loose=True)
                if la and lb and la == lb:
                    near += 1
    n = len(ref)
    return {"n_ref": n, "n_hyp": len(hyp), "sub": sub, "dele": dele, "ins": ins,
            "cer": (sub + dele + ins) / n if n else 0.0,
            "homo": homo, "near": near,
            "homo_pct": 100 * homo / sub if sub else 0.0,
            "near_pct": 100 * near / sub if sub else 0.0}
