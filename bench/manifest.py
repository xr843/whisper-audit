"""公开测试集接入。

评测与数据来源解耦：只要能给出 `{audio, text}`，就能评。
公开集因网络下不动时不阻塞任何事——本机走代理，这是已知风险。

manifest 是每行一个 json 的 jsonl：

    {"audio": "/path/BAC009S0764W0121.wav", "text": "甚至出现交易几乎停滞的情况"}
"""
import json

from audio_transcribe.evaluate import score

_SUM_KEYS = ("sub", "dele", "ins", "homo", "near", "n_ref", "n_hyp")


def read_manifest(path):
    items = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def eval_manifest(items, transcribe_fn, keep_hyps=True):
    """transcribe_fn(audio_path) -> 识别文本。

    **按字加权累加后重算 cer，不是对各条的 cer 取平均。**
    逐条平均会被短句放大、也会被 n_ref=0 的条目（cer 恒为 0）拉低，
    详见 evaluate.score 的 docstring。

    `keep_hyps=True` 时把逐条识别结果带回（hyps 字段）——转录烧的是 GPU
    小时，评分口径的任何调整都不该逼人重烧；留着 hyp 就能纯 CPU 重算。
    """
    tot = {k: 0 for k in _SUM_KEYS}
    hyps = []
    for it in items:
        hyp = transcribe_fn(it["audio"])
        if keep_hyps:
            hyps.append({"audio": it["audio"], "text": it["text"], "hyp": hyp})
        r = score(it["text"], hyp)
        for k in _SUM_KEYS:
            tot[k] += r[k]
    n, s = tot["n_ref"], tot["sub"]
    tot["n_items"] = len(items)
    tot["cer"] = (tot["sub"] + tot["dele"] + tot["ins"]) / n if n else 0.0
    tot["homo_pct"] = 100 * tot["homo"] / s if s else 0.0
    tot["near_pct"] = 100 * tot["near"] / s if s else 0.0
    if keep_hyps:
        tot["hyps"] = hyps
    return tot
