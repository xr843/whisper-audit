"""LLM 同音校订：把 LLM 关进笼子。

LLM 结合上下文改同音错字（`旧帐倒进去` → `旧账倒进去`（帐/账 同音）），但 LLM 会自由发挥、
删内容、改事实。`constrain` 是唯一的护栏：只放行同音替换，长度必须与原文
一致（否则整块拒绝），逐字比对拼音，不同音的字一律还原为原字并计数。
这样 LLM 无法删内容、无法增内容、无法换位——它能做的只有把同音字换成
同音字。
"""

import json
import urllib.error
import urllib.request

from audio_transcribe.evaluate import pinyin_key

PROMPT = (
    "你在校对中文语音识别稿。只允许把同音错别字改成正确的字。\n"
    "硬性要求：\n"
    "1. 输出字数必须与输入完全一致，一个字都不能增删\n"
    "2. 不得修改任何标点符号\n"
    "3. 只改读音相同的错别字，读音不同的一律不动\n"
    "4. 直接输出校对后的文本，不要解释、不要加引号\n"
)


def constrain(before, after, loose=False):
    """只放行同音替换。长度不一致整块拒绝。"""
    if len(after) != len(before):
        return before, {"rejected": 0, "accepted": 0, "reason": "length"}
    out, rejected, accepted = [], 0, 0
    for a, b in zip(before, after):
        if a == b:
            out.append(a)
        elif pinyin_key(a, loose) == pinyin_key(b, loose) and pinyin_key(a, loose) != ():
            out.append(b)
            accepted += 1
        else:
            out.append(a)
            rejected += 1
    return "".join(out), {"rejected": rejected, "accepted": accepted, "reason": None}


def _chunks(text, size):
    """按句末标点切，避免把一句话劈成两块。"""
    out, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= size and ch in "。！？":
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def _call(base_url, model, api_key, text, timeout=120):
    body = json.dumps({
        "model": model, "temperature": 0,
        "messages": [{"role": "system", "content": PROMPT},
                     {"role": "user", "content": text}],
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def polish(text, *, base_url, model, api_key, chunk=1200, dry_run=False,
           loose=False):
    parts = _chunks(text, chunk)
    rep = {"chunks": len(parts), "accepted": 0, "rejected": 0,
           "length_rejected": 0, "failed": 0}
    if dry_run:
        print(f"[dry-run] 将发送 {len(parts)} 块，共 {len(text)} 字")
        print(f"[dry-run] 首块内容：\n{parts[0][:300]}")
        return text, rep

    out = []
    for p in parts:
        try:
            got = _call(base_url, model, api_key, p)
        except (urllib.error.URLError, KeyError, TimeoutError):
            rep["failed"] += 1          # 网络失败保留原文，不中断整体流程
            out.append(p)
            continue
        fixed, r = constrain(p, got, loose)
        if r["reason"] == "length":
            rep["length_rejected"] += 1
        rep["accepted"] += r["accepted"]
        rep["rejected"] += r["rejected"]
        out.append(fixed)
    return "".join(out), rep
