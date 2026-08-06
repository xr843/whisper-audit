# 正确率可测量化、提升与开源化 —— 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 audio-transcribe 建立字级 CER 评测与回归门禁，接入三条正确率提升路径（拼音术语纠错、中文专用第二引擎、LLM 同音校订），并完成开源化。

**Architecture:** 先把 849 行的 `transcribe.py` 拆成 `audio_transcribe/` 包，ASR 后端抽象为 Engine 接口。评测模块独立于流水线，纯文本输入输出，因此不需要 GPU 也不需要音频即可在 CI 跑。三条提升路径各自是可开关的独立阶段，每条都必须拿出独立的 CER 增量数字才允许默认开启。

**Tech Stack:** Python 3.12 / faster-whisper / FunASR / rapidfuzz / pypinyin / opencc / pytest / GitHub Actions

## Global Constraints

- 默认档速度退化不得超过 5%，基线 **24.5x**，下限 23.3x（bench15.wav，词级时间戳开）
  （15 分钟片段、int8_float16、batch 16、beam 5、**词级时间戳开启**）。
  基线必须测流水线实际跑的配置——词级时间戳现在是默认开的，拿关闭时的数字卡门禁
  等于在测一个不存在的配置
- 核心依赖必须是纯 CPU 可安装的（opencc / pypinyin / rapidfuzz / numpy），ASR 后端走 optional-dependencies，否则 CI 装不动
- `python3 transcribe.py 录音.mp3` 的行为必须保持不变
- 现有 24 个测试全程保持通过，不允许为了让新代码过关而修改既有断言
- 任何「自动改动或删除正文」的逻辑必须有锁死测试，改动必须记账进质检报告
- 每项提升测不出 CER 收益就默认关闭，不是「装上了就开着」
- 提交信息用中文，不带任何 AI 署名 trailer
- git 身份：`xr843 <137012659+xr843@users.noreply.github.com>`

## 阶段与并行

| 阶段 | 任务 | 并行性 |
|---|---|---|
| 一、地基 | Task 1–2 | **必须串行且最先**，其余全部依赖它 |
| 二、尺子 | Task 3 → 4、4b、5 | Task 3 先行；4 / 4b / 5 都依赖它，三者之间可并行 |
| 三、提升 | Task 6→7、8→9、10→11 | 三条链**互相独立可并行**，每条链内部串行 |
| 四、收尾 | Task 12、13 | Task 13 可与阶段二、三并行；Task 12 须在 9、11 之后 |

**并行分派建议**（你已批准多开 agent）：

1. Task 1、2 我自己串行做完——拆包动全部文件，并行必冲突
2. 然后同时开 **4 个 agent**：Task 3（评测核心）、Task 6（拼音纠错）、Task 8（FunASR 适配）、Task 13（开源化）
   - 这四个只碰各自的新文件，唯一交集是 Task 6 依赖 Task 3 的 `pinyin_key`——
     所以给 Task 6 的 agent 明确写死该函数签名，它不必等
3. Task 3 完成后再开 2 个：Task 4（金标）、Task 4b（manifest）
4. Task 10（polish 核心）随时可开，它只依赖 `pinyin_key`
5. 接入类任务 7、9、11 都改 `cli.py`，**必须串行**，我自己按序做

**冲突面**：`cli.py` 是唯一的热点文件，被 Task 2/7/9/11/13 触碰。所有涉及它的任务一律串行。

---

## Task 1: 拆包骨架

**Files:**
- Create: `audio_transcribe/__init__.py`, `audio_transcribe/audio.py`, `audio_transcribe/audit.py`, `audio_transcribe/merge.py`, `audio_transcribe/render.py`, `audio_transcribe/cli.py`, `audio_transcribe/engines/__init__.py`, `audio_transcribe/engines/whisper.py`
- Modify: `transcribe.py`（改为薄入口）, `tests/test_pipeline.py`（改 import）

**Interfaces:**
- Consumes: 无
- Produces:
  - `audio_transcribe.audio`: `ensure_cuda_libs()`, `prepare_audio(src, workdir) -> str`, `class Loudness(wav)` 带 `.db(t0,t1)` `.close()`
  - `audio_transcribe.audit`: `SPEECH_RATE=3.0`, `HALLU_PAT`, `starved_spans(segments, min_len=15.0, min_density=1.2, max_span=90.0) -> list[[float,float,str]]`, `audit(data, loud, gap_min=3.0) -> dict`, `audit_rows(rows, dur, min_len=15.0, min_density=1.2) -> dict`, `find_breaks(data, loud, min_len=120, step=10, max_n=3) -> list[tuple]`, `in_any(t, spans, pad=0.0) -> bool`
  - `audio_transcribe.merge`: `strip_common(a,b,min_block=8) -> str`, `merge_rows(rows) -> list[dict]`, `combine(passes, patch, terms, breaks, dur, drop_spans=()) -> list[dict]`
  - `audio_transcribe.render`: `DEFAULT_LEAD`, `CLAUSE_GLUE`, `insert_clause_breaks(t, lead=None, min_run=16, max_run=38) -> str`, `gap_thresholds(rows, lo=0.25, hi=0.60) -> tuple[float,float]`, `punctuate_row(row, comma_gap=0.35, period_gap=0.9) -> str`, `resplit_rows(rows, max_dur=8.0, max_chars=30) -> list[dict]`, `raw_text(passes) -> str`, `terms_hits(text, terms) -> dict`, `render(rows, dur, outdir, title, terms, meta) -> tuple[int,int,int]`
  - `audio_transcribe.engines`: `class Engine` 抽象基类（`name: str`、`transcribe(wav, **opts) -> dict`）、`get_engine(name, **kw) -> Engine`
  - `audio_transcribe.engines.whisper`: `class WhisperEngine(Engine)`，`name = "whisper"`
  - `audio_transcribe.cli`: `main(argv=None) -> int`

- [ ] **Step 1: 建包目录与空模块**

```bash
mkdir -p audio_transcribe/engines
touch audio_transcribe/__init__.py audio_transcribe/engines/__init__.py
```

- [ ] **Step 2: 按职责搬运现有代码，不改任何逻辑**

把 `transcribe.py` 现有内容原样切分到各模块。切分边界严格按现有的注释分隔线：

| 现有分隔线 | 去向 |
|---|---|
| `ensure_cuda_libs` / `prepare_audio` / `class Loudness` | `audio.py` |
| `PROFILES` / `transcribe_pass` / `repatch` | `engines/whisper.py`（`PROFILES` 留 `cli.py`） |
| `HALLU_*` / `SPEECH_RATE` / `starved_spans` / `audit_rows` / `audit` / `find_breaks` / `in_any` | `audit.py` |
| `strip_common` / `merge_rows` / `combine` | `merge.py` |
| `DEFAULT_LEAD` / `CLAUSE_GLUE` / `insert_clause_breaks` / `gap_thresholds` / `punctuate_row` / `resplit_rows` / `_split_row` / `raw_text` / `terms_hits` / `render` | `render.py` |
| `log` / `hms` | `audio_transcribe/__init__.py` |
| `main()` | `cli.py` |

搬运时唯一允许的改动是补 import。**逻辑一行都不许改。**

- [ ] **Step 3: 定义 Engine 抽象**

`audio_transcribe/engines/__init__.py`:

```python
"""ASR 后端抽象。

引入第二个引擎的目的是跨模型互补——实测同模型不同 chunk 的双路净贡献只有
1,792 字（全文 4%），却要付 +31 分钟。跨引擎的互补性应当远大于此，但这是
待验证的假设，由 CER 裁决。
"""


class Engine:
    name = "base"

    def transcribe(self, wav, **opts):
        """返回 {"duration": float, "segments": [...]}。

        segment 必须有 start/end/text；avg_logprob、no_speech_prob、words 可缺。
        words 缺失时下游自动降级：字幕退回整段、段内标点不加。
        """
        raise NotImplementedError


_REGISTRY = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_engine(name, **kw):
    if name not in _REGISTRY:
        raise KeyError(f"未知引擎 {name}，已注册：{sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)
```

`audio_transcribe/engines/whisper.py` 里把现有 `transcribe_pass` 包成类：

```python
from . import Engine, register


@register
class WhisperEngine(Engine):
    name = "whisper"

    def __init__(self, model_name="large-v3", device="cuda", compute="int8_float16"):
        self.model_name, self.device, self.compute = model_name, device, compute

    def transcribe(self, wav, out_json=None, chunk_length=30, batch=16, beam=5,
                   language="zh", **_):
        return transcribe_pass(wav, out_json, chunk_length, batch, beam,
                               self.compute, self.model_name, self.device, language)
```

- [ ] **Step 4: `transcribe.py` 改为薄入口**

```python
#!/usr/bin/env python3
"""长音频转录流水线 —— 兼容入口。实现在 audio_transcribe/ 包里。

    python3 transcribe.py 录音.mp3 -o 输出目录 --profile meeting
"""
import sys

from audio_transcribe.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 改测试的 import 为直接模块引用**

`tests/test_pipeline.py` 去掉 `import transcribe as T`，改成：

```python
from audio_transcribe import audit, merge, render
```

再把测试体里的 `T.xxx` 逐个替换为它真正所属的模块：

| 原引用 | 改为 |
|---|---|
| `T.starved_spans` / `T.audit` / `T.audit_rows` | `audit.xxx` |
| `T.merge_rows` / `T.combine` / `T.strip_common` | `merge.xxx` |
| `T.insert_clause_breaks` / `T.gap_thresholds` / `T.punctuate_row` / `T.resplit_rows` / `T.terms_hits` | `render.xxx` |

**只改引用路径，断言表达式一个字都不许动。** 不要用 `types.SimpleNamespace` 之类的
聚合别名把三个模块糊成一个 `T`——那是测试反模式，而且开源后别人读到会困惑。

- [ ] **Step 6: 跑测试确认 24 个全过**

Run: `python3 -m pytest tests/ -q`
Expected: `24 passed`

- [ ] **Step 7: 跑真机冒烟确认行为没变**

Run: `python3 transcribe.py <5分钟音频> -o /tmp/t1 --profile lecture --title t1`
Expected: 正常输出 4 个文件，日志里有「终审：合并稿覆盖 …」

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "拆包：transcribe.py 拆为 audio_transcribe/，ASR 后端抽象为 Engine 接口"
```

---

## Task 2: 打包与命令行入口

**Files:**
- Create: `pyproject.toml`
- Modify: `audio_transcribe/cli.py`, `requirements.txt`

**Interfaces:**
- Consumes: `audio_transcribe.cli.main`
- Produces: 命令 `audio-transcribe`，子命令 `run`（默认）/ `eval` / `goldset`

- [ ] **Step 1: 写 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "audio-transcribe"
version = "0.2.0"
description = "长音频转文档流水线，以不遗漏为目标，带覆盖率审计与字级 CER 评测"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "opencc-python-reimplemented>=0.1.7",
    "pypinyin>=0.53",
    "rapidfuzz>=3.9",
    "numpy>=1.24",
]

[project.optional-dependencies]
whisper = ["faster-whisper>=1.1.0", "ctranslate2>=4.5,<5"]
funasr = ["funasr>=1.1.0"]
dev = ["pytest>=8.0"]

[project.scripts]
audio-transcribe = "audio_transcribe.cli:main"

[tool.setuptools.packages.find]
include = ["audio_transcribe*"]
```

核心依赖必须纯 CPU 可装，ASR 后端走 extras——否则 CI 装不动。

- [ ] **Step 2: cli.py 改成子命令结构**

`main(argv=None)` 用 `argparse` 的 subparsers。**没写子命令时默认走 `run`**，保证 `python3 transcribe.py 录音.mp3` 不变：

```python
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("run", "eval", "goldset", "-h", "--help"):
        argv.insert(0, "run")
    ap = argparse.ArgumentParser(prog="audio-transcribe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_run_args(sub.add_parser("run", help="转录"))
    ...
```

- [ ] **Step 3: 装包并验证命令可用**

Run:
```bash
pip install --user --break-system-packages -e .
audio-transcribe --help
python3 transcribe.py --help
```
Expected: 两条都打印用法，无 traceback

- [ ] **Step 4: 跑测试**

Run: `python3 -m pytest tests/ -q`
Expected: `24 passed`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "打包：pyproject.toml 与 audio-transcribe 命令行入口"
```

---

## Task 3: CER 评测核心（可并行）

**Files:**
- Create: `audio_transcribe/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `normalize(text) -> str`
  - `pinyin_key(text, loose=False) -> tuple[str, ...]`
  - `score(ref, hyp) -> dict`，键为 `n_ref` `n_hyp` `sub` `dele` `ins` `cer` `homo` `near` `homo_pct` `near_pct`

`homo` = 替换错误里拼音**严格相同**的个数；`near` = 拼音**近似**的个数（含 `homo`）。
近似规则覆盖方言口音常见混淆：`zh/z`、`ch/c`、`sh/s`、`n/l`、以及 `ang/an`、`eng/en`、`ing/in`。

- [ ] **Step 1: 写失败测试**

`tests/test_evaluate.py`:

```python
import pytest

from audio_transcribe import evaluate as E


def test_normalize_strips_punctuation_and_converts_traditional():
    assert E.normalize("上課的時間，差不多到了。") == "上课的时间差不多到了"


def test_normalize_unifies_digits():
    assert E.normalize("第9条") == E.normalize("第九条")


def test_score_counts_substitution_deletion_insertion():
    r = E.score("甲乙丙丁", "甲X丙丁戊")      # 乙→X 替换，末尾多一个戊
    assert r["sub"] == 1
    assert r["ins"] == 1
    assert r["dele"] == 0
    assert r["cer"] == pytest.approx(0.5, abs=1e-9)


def test_score_deletion_is_reported_separately():
    """删除率是「不遗漏」这个卖点的直接度量，必须单列。"""
    r = E.score("甲乙丙丁戊己", "甲乙丙")
    assert r["dele"] == 3
    assert r["sub"] == 0


def test_homophone_substitution_is_measured():
    """旧账→旧帐：账/帐 同音，这类错拼音纠错能修。"""
    r = E.score("旧账导进去", "旧帐导进去")
    assert r["sub"] == 1
    assert r["homo"] == 1


def test_near_homophone_catches_accent_confusion():
    """方言口音把 zh 读成 z：账(zhang)→赃(zang) 不同音但近音。"""
    r = E.score("旧账", "旧赃")
    assert r["sub"] == 1
    assert r["homo"] == 0
    assert r["near"] == 1


def test_unrelated_error_is_neither_homophone_nor_near():
    r = E.score("旧账", "旧猫")
    assert r["sub"] == 1
    assert r["homo"] == 0
    assert r["near"] == 0


def test_perfect_match_scores_zero():
    r = E.score("完全一样的文本", "完全一样的文本")
    assert r["cer"] == 0.0
    assert r["homo_pct"] == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/test_evaluate.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'audio_transcribe.evaluate'`

- [ ] **Step 3: 实现 evaluate.py**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest tests/test_evaluate.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add audio_transcribe/evaluate.py tests/test_evaluate.py
git commit -m "评测：字级 CER 与增删改分解，单列删除率与同音/近音替换率"
```

---

## Task 4: 金标生成与评测命令

**Files:**
- Create: `audio_transcribe/goldset.py`, `tests/test_goldset.py`
- Modify: `audio_transcribe/cli.py`

**Interfaces:**
- Consumes: `audio_transcribe.evaluate.score`
- Produces:
  - `make_goldset(srt_path, start=None, end=None) -> list[tuple[float, str]]`
  - `write_tsv(rows, path)` / `read_tsv(path) -> list[tuple[float, str]]`
  - `evaluate_dir(gold_path, hyp_srt_path) -> dict`
  - CLI：`audio-transcribe goldset <输出目录> --from 00:10:00 --to 00:25:00 -o sample.gold.tsv`
  - CLI：`audio-transcribe eval --gold sample.gold.tsv --hyp <输出目录>`

**金标格式**：两列 TSV，`秒数<TAB>文本`。人工只改错字，不碰格式、不打时间戳。

- [ ] **Step 1: 写失败测试**

```python
import io

from audio_transcribe import goldset as G

SRT = """1
00:00:01,000 --> 00:00:03,000
上课的时间差不多到了

2
00:00:03,500 --> 00:00:06,000
请外面的学员进场入座

3
00:10:00,000 --> 00:10:04,000
这一条在时间窗之外
"""


def test_make_goldset_extracts_rows(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text(SRT, encoding="utf-8")
    rows = G.make_goldset(str(p))
    assert len(rows) == 3
    assert rows[0] == (1.0, "上课的时间差不多到了")


def test_make_goldset_respects_time_window(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text(SRT, encoding="utf-8")
    rows = G.make_goldset(str(p), start=0.0, end=10.0)
    assert len(rows) == 2


def test_tsv_roundtrip(tmp_path):
    rows = [(1.0, "甲乙丙"), (3.5, "丁戊己")]
    p = tmp_path / "g.tsv"
    G.write_tsv(rows, str(p))
    assert G.read_tsv(str(p)) == rows


def test_tsv_tolerates_hand_edited_whitespace(tmp_path):
    """人工改完可能留下多余空格或空行，读取必须容忍。"""
    p = tmp_path / "g.tsv"
    p.write_text("1.0\t 甲乙丙 \n\n3.5\t丁戊己\n", encoding="utf-8")
    assert G.read_tsv(str(p)) == [(1.0, "甲乙丙"), (3.5, "丁戊己")]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/test_goldset.py -q`
Expected: FAIL，`No module named 'audio_transcribe.goldset'`

- [ ] **Step 3: 实现 goldset.py**

```python
"""金标生成与评测入口。

格式是两列 TSV：秒数<TAB>文本。初稿直接取 ASR 结果，
人工只改错字——不碰格式、不打时间戳、不做对齐。
"""
import re

from .evaluate import score

_CUE = re.compile(r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)\s*-->\s*"
                  r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)")


def _sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


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
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        t, _, text = line.partition("\t")
        rows.append((float(t.strip()), text.strip()))
    return rows


def evaluate_dir(gold_path, hyp_srt_path):
    gold = read_tsv(gold_path)
    if not gold:
        raise ValueError(f"金标为空：{gold_path}")
    lo = min(t for t, _ in gold)
    hi = max(t for t, _ in gold)
    ref = "".join(text for _, text in gold)
    hyp = "".join(r["text"] for r in parse_srt(hyp_srt_path)
                  if lo <= r["start"] <= hi)
    return score(ref, hyp)
```

> 注意 `evaluate_dir` 的时间窗上界取金标最后一条的 start，因此金标末尾那条对应的
> hyp 内容会被包含（`<=`）。若实测发现末尾漏字，改为 `hi = max(...) + 最后一条时长`
> 并补一个测试锁住。

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest tests/test_goldset.py -q`
Expected: `4 passed`

- [ ] **Step 5: 接 CLI 两个子命令，人工验证**

Run:
```bash
audio-transcribe goldset /tmp/t1 --from 00:00:00 --to 00:05:00 -o /tmp/t1.gold.tsv
head -3 /tmp/t1.gold.tsv
audio-transcribe eval --gold /tmp/t1.gold.tsv --hyp /tmp/t1
```
Expected: TSV 两列可读；eval 打印 CER 表格（未改错字时 CER 应为 0.000）

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "金标：goldset 生成待校对稿，eval 子命令跑 CER"
```

---

## Task 4b: 公开集 manifest 与获取脚本

**Files:**
- Create: `bench/manifest.py`, `bench/fetch_aishell_test.sh`, `tests/test_manifest.py`
- Modify: `audio_transcribe/cli.py`（`eval` 增加 `--manifest`）

**Interfaces:**
- Consumes: `audio_transcribe.evaluate.score`
- Produces: `read_manifest(path) -> list[dict]`、`eval_manifest(items, transcribe_fn) -> dict`

manifest 是每行 `{"audio": "...", "text": "..."}` 的 jsonl。评测与数据来源解耦，
公开集因网络下不动时不阻塞任何事——这是本机走代理的已知风险。

- [ ] **Step 1: 写失败测试**

```python
import json

from bench import manifest as M


def test_read_manifest_skips_blank_lines(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"audio":"a.wav","text":"甲乙"}\n\n{"audio":"b.wav","text":"丙丁"}\n',
                 encoding="utf-8")
    items = M.read_manifest(str(p))
    assert [i["text"] for i in items] == ["甲乙", "丙丁"]


def test_eval_manifest_aggregates_over_items():
    items = [{"audio": "a.wav", "text": "旧账导进去"},
             {"audio": "b.wav", "text": "会计凭证"}]
    fake = {"a.wav": "旧帐导进去", "b.wav": "会计凭证"}
    rep = M.eval_manifest(items, lambda p: fake[p])
    assert rep["n_items"] == 2
    assert rep["sub"] == 1
    assert rep["homo"] == 1
    assert rep["cer"] > 0


def test_eval_manifest_on_perfect_hypotheses_is_zero():
    items = [{"audio": "a.wav", "text": "甲乙丙"}]
    rep = M.eval_manifest(items, lambda p: "甲乙丙")
    assert rep["cer"] == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/test_manifest.py -q`
Expected: FAIL，`No module named 'bench'`

- [ ] **Step 3: 实现 bench/manifest.py**

```python
"""公开测试集接入。评测与数据来源解耦：只要能给出 {audio, text}，就能评。"""
import json

from audio_transcribe.evaluate import score


def read_manifest(path):
    items = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def eval_manifest(items, transcribe_fn):
    """transcribe_fn(audio_path) -> 识别文本。整体按字加权汇总，不是逐条平均。"""
    tot = {"sub": 0, "dele": 0, "ins": 0, "homo": 0, "near": 0, "n_ref": 0}
    for it in items:
        r = score(it["text"], transcribe_fn(it["audio"]))
        for k in tot:
            tot[k] += r[k]
    n = tot["n_ref"]
    tot["n_items"] = len(items)
    tot["cer"] = (tot["sub"] + tot["dele"] + tot["ins"]) / n if n else 0.0
    tot["homo_pct"] = 100 * tot["homo"] / tot["sub"] if tot["sub"] else 0.0
    tot["near_pct"] = 100 * tot["near"] / tot["sub"] if tot["sub"] else 0.0
    return tot
```

需要 `bench/__init__.py`（空文件）让它可 import。

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest tests/test_manifest.py -q`
Expected: `3 passed`

- [ ] **Step 5: 写获取脚本**

`bench/fetch_aishell_test.sh`：从 openslr 拉 AISHELL-1，只解出 test 部分，
生成 `bench/aishell_test.jsonl`。脚本必须做到：

- 下载失败时打印明确原因并以非零退出，**不留半个文件**
- 已存在时跳过重下
- 最后打印生成了多少条

```bash
#!/usr/bin/env bash
# 获取 AISHELL-1 test 作为公开基准。本机走代理，下载可能失败——失败不阻塞项目。
set -euo pipefail
DIR="${1:-bench/data/aishell}"
mkdir -p "$DIR"
if [ -f "$DIR/../aishell_test.jsonl" ]; then echo "已存在，跳过"; exit 0; fi
URL="https://openslr.magicdatatech.com/resources/33/data_aishell.tgz"
echo "下载 $URL （约 15GB，可中断续传）"
curl -fL --retry 3 -C - -o "$DIR/data_aishell.tgz" "$URL"
echo "下载完成，解包 test 部分…"
# 解包与 transcript 对齐的具体步骤见 bench/README.md
```

> 实施者：AISHELL-1 是 15GB 全量包，若带宽或代理不允许，**不要卡在这里**。
> 在 `bench/README.md` 记录失败原因，把公开集这条腿标为「未接入」，
> 项目其余部分不依赖它。真实 CER 以自建金标为准。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "评测：manifest 驱动的公开集接入与 AISHELL-1 获取脚本"
```

---

## Task 5: CI 回归夹具与门禁

**Files:**
- Create: `tests/fixtures/regression.jsonl`, `tests/test_regression.py`

**Interfaces:**
- Consumes: `audio_transcribe.evaluate.score`
- Produces: 固定的 `(ref, hyp)` 文本对与期望指标，作为 CI 门禁

CI 跑不了 ASR（无 GPU、无音频），所以门禁只能盯**纯文本阶段**：评测工具本身的输出稳定性，
以及后续拼音纠错 / polish 在固定输入上的表现。这是诚实的边界，不假装覆盖了 ASR。

- [ ] **Step 1: 造夹具**

`tests/fixtures/regression.jsonl`，每行 `{"id","ref","hyp"}`，取自真实错误模式（内容为构造，不含敏感信息）：

```json
{"id":"homophone","ref":"旧账导进去明细账导进去","hyp":"旧帐导进去明细帐导进去"}
{"id":"near_accent","ref":"大型活动场所财务管理","hyp":"大型活动厂所财务管理"}
{"id":"deletion","ref":"应当指定三人管理开启时三人同时在场","hyp":"应当指定三人管理"}
{"id":"clean","ref":"会计凭证包括记账凭证和原始凭证","hyp":"会计凭证包括记账凭证和原始凭证"}
```

- [ ] **Step 2: 写测试**

```python
import json
import pathlib

from audio_transcribe import evaluate as E

FIX = pathlib.Path(__file__).parent / "fixtures" / "regression.jsonl"


def load():
    return [json.loads(l) for l in FIX.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_fixture_metrics_are_stable():
    got = {r["id"]: E.score(r["ref"], r["hyp"]) for r in load()}
    assert got["clean"]["cer"] == 0.0
    assert got["deletion"]["dele"] == 9 and got["deletion"]["sub"] == 0
    assert got["homophone"]["homo"] == 2
    assert got["near_accent"]["near"] == 1 and got["near_accent"]["homo"] == 1
```

> 实施者注意：`deletion` 的 `dele` 与 `near_accent` 的具体数字请先跑一次实际输出核对，
> 若与此处不符，说明规范化或近音规则与预期不同——**先查清原因再改数字**，
> 不允许直接把断言改成实际输出。

- [ ] **Step 3: 跑测试**

Run: `python3 -m pytest tests/test_regression.py -q`
Expected: `1 passed`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "CI 夹具：固定文本对的 CER 回归门禁"
```

---

## Task 6: 拼音级术语纠错（可并行）

**Files:**
- Create: `audio_transcribe/terms.py`, `tests/test_terms.py`
- Modify: `examples/terms/finance-lecture.json`（Task 13 建，此处先用测试内联数据）

**Interfaces:**
- Consumes: `audio_transcribe.evaluate.pinyin_key`
- Produces: `pinyin_fix(text, terms, loose=True) -> tuple[str, list[dict]]`
  - 返回 `(修正后文本, 命中列表)`，命中项为 `{"pos": int, "from": str, "to": str}`
  - 术语表新增 `terms` 字段：只列**正确写法**的字符串列表

现有术语表是字面替换，54 条里 12 条从未命中——误识写法穷举不完。
一条正确写法配拼音匹配，能覆盖一整类同音错。

- [ ] **Step 1: 写失败测试**

```python
import pytest

from audio_transcribe import terms as T

TERMS = {"terms": ["财税管理", "大型活动场所", "非营利组织", "记账凭证"]}


def test_exact_homophone_is_corrected():
    out, hits = T.pinyin_fix("才税管理的要点", TERMS)
    assert out == "财税管理的要点"
    assert hits[0]["from"] == "才税管理" and hits[0]["to"] == "财税管理"


def test_near_homophone_is_corrected_when_loose():
    """厂所(chang suo) vs 场所(chang suo)——同音。"""
    out, _ = T.pinyin_fix("大型活动厂所登记", TERMS)
    assert out == "大型活动场所登记"


def test_correct_text_is_left_alone():
    out, hits = T.pinyin_fix("非营利组织的记账凭证", TERMS)
    assert out == "非营利组织的记账凭证"
    assert hits == []


def test_different_pinyin_is_never_touched():
    """锁死：拼音不同的词一律不许改动。这是防止静悄悄改内容的护栏。"""
    src = "会计科目和固定资产的处理"
    out, hits = T.pinyin_fix(src, TERMS)
    assert out == src
    assert hits == []


def test_longer_term_wins_over_shorter():
    terms = {"terms": ["场所", "大型活动场所"]}
    out, _ = T.pinyin_fix("大型活动厂所", terms)
    assert out == "大型活动场所"


def test_hits_record_position_for_accounting():
    """所有改动必须可追溯，写进质检报告。"""
    _, hits = T.pinyin_fix("前面才税管理后面", TERMS)
    assert hits[0]["pos"] == 2


def test_empty_terms_is_noop():
    assert T.pinyin_fix("任何文本", {}) == ("任何文本", [])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/test_terms.py -q`
Expected: FAIL，`No module named 'audio_transcribe.terms'`

- [ ] **Step 3: 实现 terms.py**

```python
"""术语表：字面精确替换 + 拼音模糊匹配。

字面替换（fixes）穷举不完误识写法——实测 54 条里 12 条从未命中。
拼音匹配只需列正确写法，一条覆盖一整类同音错。

护栏：只在拼音一致（或近音，可关）时替换，所有改动记账。
拼音不同的词一律不许碰——这是这个项目栽过两次的坑（静悄悄改内容）。
"""
from .evaluate import pinyin_key


def pinyin_fix(text, terms, loose=True):
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest tests/test_terms.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add audio_transcribe/terms.py tests/test_terms.py
git commit -m "术语表：拼音级模糊纠错，只在同音/近音时替换并全量记账"
```

---

## Task 7: 接入拼音纠错并测出 CER 增量

**Files:**
- Modify: `audio_transcribe/merge.py`（`combine` 的 `norm`）, `audio_transcribe/cli.py`
- Create: `tests/test_pinyin_integration.py`

**Interfaces:**
- Consumes: `terms.pinyin_fix`
- Produces: 质检报告新增 `pinyin_fixes: list[dict]`；CLI 新增 `--pinyin-fix`（**默认关**，实施时据实测结论反转了方向）

- [ ] **Step 1: 写失败测试**

```python
from audio_transcribe import merge as M


def test_combine_applies_pinyin_fix_and_records_hits():
    passes = [{"duration": 60.0, "segments": [
        {"start": 0.0, "end": 10.0, "text": "才税管理的要点", "avg_logprob": -0.3}]}]
    terms = {"terms": ["财税管理"]}
    rows, hits = M.combine(passes, [], terms, [], 60.0, return_hits=True)
    assert "财税管理" in "".join(r["text"] for r in rows)
    assert hits and hits[0]["to"] == "财税管理"


def test_pinyin_fix_can_be_disabled():
    passes = [{"duration": 60.0, "segments": [
        {"start": 0.0, "end": 10.0, "text": "才税管理的要点", "avg_logprob": -0.3}]}]
    rows, hits = M.combine(passes, [], {"terms": ["财税管理"]}, [], 60.0,
                           pinyin=False, return_hits=True)
    assert "才税管理" in "".join(r["text"] for r in rows)
    assert hits == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/test_pinyin_integration.py -q`
Expected: FAIL，`combine() got an unexpected keyword argument 'return_hits'`

- [ ] **Step 3: 实现**

`merge.py` 顶部加 `from .terms import pinyin_fix`，`combine` 签名与 `norm` 改为：

```python
def combine(passes, patch, terms, breaks, dur, drop_spans=(),
            pinyin=True, return_hits=False):
    """逐 30 秒窗口在各路之间取字数多的，再用补转填两路都空的洞。"""
    import opencc
    cc = opencc.OpenCC("t2s")
    fixes = terms.get("fixes", [])
    hits = []

    def norm(t):
        t = cc.convert(t).strip()
        for a, b in fixes:              # 字面替换优先：人工逐条确认过的
            t = t.replace(a, b)
        if pinyin:
            t, h = pinyin_fix(t, terms)
            hits.extend(h)
        return t
```

函数末尾：

```python
    rows = merge_rows(picked)
    return (rows, hits) if return_hits else rows
```

> 注意 `norm()` 在 `combine` 里对同一段文本会被调用多次（`usable()` 判定时一次、
> 构造 `prepped` 时一次），命中会重复累积。**去重后再写报告**：
> `hits = list({(h["pos"], h["from"], h["to"]): h for h in hits}.values())`。

`cli.py` 改调用点并记账：

```python
rows, pyhits = combine(passes, patch, terms, breaks, dur, rep.get("drop", []),
                       pinyin=not args.no_pinyin_fix, return_hits=True)
if pyhits:
    log(f"拼音纠错：{len(pyhits)} 处（例：{pyhits[0]['from']}→{pyhits[0]['to']}）")
```

质检报告的 json 里加 `"pinyin_fixes": pyhits`。

- [ ] **Step 4: 跑全部测试**

Run: `python3 -m pytest tests/ -q`
Expected: 全绿（24 + 新增）

- [ ] **Step 5: 测出真实 CER 增量**

Run:
```bash
audio-transcribe eval --gold <金标>.gold.tsv --hyp <开启纠错的输出目录>
audio-transcribe eval --gold <金标>.gold.tsv --hyp <关闭纠错的输出目录>
```
把两个 CER 与 `homo_pct` 记进 `docs/measurements.md`。
**判定（实施后修订）：默认关闭。** 不只看 CER——存在系统性误伤（节余/结余这类真实词被判同音）就必须默认关，哪怕 CER 略有改善。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "接入拼音纠错：改动全量记账进质检报告，--pinyin-fix 可关"
```

---

## Task 8: FunASR 第二引擎（可并行）

**Files:**
- Create: `audio_transcribe/engines/funasr.py`, `tests/test_funasr_adapter.py`

**Interfaces:**
- Consumes: `audio_transcribe.engines.Engine` / `register`
- Produces: `class FunASREngine(Engine)`，`name = "funasr"`；`to_segments(raw) -> list[dict]`

- [ ] **Step 1: 先探明 FunASR 的实际输出格式，不要凭记忆写适配**

Run:
```bash
pip install --user --break-system-packages funasr
python3 -c "
from funasr import AutoModel
m = AutoModel(model='paraformer-zh', vad_model='fsmn-vad', device='cuda')
r = m.generate(input='<5分钟音频>.wav', batch_size_s=300, sentence_timestamp=True)
import json; print(json.dumps(r, ensure_ascii=False)[:1500])
"
```
Expected: 打印实际结构。**把结构记在 `funasr.py` 的文档字符串里**，适配器按实际结构写。
若模型下载失败，记录失败信息并跳到 Step 6（引擎可选，缺失时降级）。

- [ ] **Step 2: 写适配器的失败测试**

**必须用 Step 1 抓到的真实结构重写下面的夹具。** 下例是按 FunASR 文档推测的形状，
未经实测；如果 Step 1 打印出来的字段名不同（例如是 `timestamp` 而非 `sentence_info`，
或时间单位是秒而非毫秒），**以实测为准改测试，而不是改适配器去迁就这段推测**：

```python
from audio_transcribe.engines import funasr as F


def test_to_segments_maps_sentence_info():
    raw = [{"text": "甲乙丙丁",
            "sentence_info": [
                {"text": "甲乙", "start": 0, "end": 1200},
                {"text": "丙丁", "start": 1500, "end": 2600}]}]
    segs = F.to_segments(raw)
    assert [s["text"] for s in segs] == ["甲乙", "丙丁"]
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 1.2


def test_to_segments_tolerates_missing_timestamps():
    """没有句级时间戳时必须降级而不是崩——words 缺失下游会自动退化。"""
    segs = F.to_segments([{"text": "甲乙丙丁"}])
    assert len(segs) == 1 and segs[0]["words"] == []
```

- [ ] **Step 3: 跑测试确认失败**

Run: `python3 -m pytest tests/test_funasr_adapter.py -q`
Expected: FAIL

- [ ] **Step 4: 实现适配器与引擎类**

`to_segments` 是纯函数（毫秒→秒、字段改名、`words` 缺失填 `[]`），因此可在 CI 无 GPU 测试。
`FunASREngine.transcribe` 只负责调模型再交给 `to_segments`。

- [ ] **Step 5: 跑测试确认通过**

Run: `python3 -m pytest tests/test_funasr_adapter.py -q`
Expected: `2 passed`

- [ ] **Step 6: 真机验证引擎可跑**

Run: `python3 -c "from audio_transcribe.engines import get_engine; print(get_engine('funasr').transcribe('<5分钟音频>.wav')['segments'][:2])"`
Expected: 打印两条 segment。若模型下载失败，在文档里记录并标注此引擎不可用。

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "引擎：接入 FunASR，适配器为纯函数可离线测试"
```

---

## Task 9: 档位支持引擎组合，跨引擎合并验证

**Files:**
- Modify: `audio_transcribe/cli.py`（`PROFILES`）, `audio_transcribe/merge.py`
- Create: `tests/test_cross_engine_merge.py`

**Interfaces:**
- Produces: `PROFILES` 中每档新增 `engines: list[dict]`，如
  `[{"name":"whisper","chunk":30}, {"name":"funasr"}]`

- [ ] **Step 1: 写失败测试**

```python
from audio_transcribe import merge as M


def test_cross_engine_merge_keeps_content_unique_to_each():
    """跨引擎分段差异更大，不能因为「看起来重复」就丢掉另一路独有的内容。"""
    a = {"duration": 60.0, "segments": [
        {"start": 0.0, "end": 30.0, "text": "现金箱应当指定三人管理"}]}
    b = {"duration": 60.0, "segments": [
        {"start": 0.0, "end": 12.0, "text": "现金箱应当指定三人管理"},
        {"start": 12.0, "end": 30.0, "text": "开启时三人同时在场当场清点登记签字"}]}
    rows = M.combine([a, b], [], {}, [], 60.0)
    joined = "".join(r["text"] for r in rows)
    assert "开启时三人同时在场" in joined
    assert joined.count("现金箱应当指定三人管理") == 1


def test_engine_without_words_does_not_break_downstream():
    a = {"duration": 30.0, "segments": [
        {"start": 0.0, "end": 10.0, "text": "没有词级时间戳的引擎"}]}
    rows = M.combine([a], [], {}, [], 30.0)
    assert rows and rows[0].get("words", []) == []
```

- [ ] **Step 2: 跑测试确认失败或通过**

Run: `python3 -m pytest tests/test_cross_engine_merge.py -q`
Expected: 若现有 `combine` 已能处理则直接通过——**通过也不许跳过本任务**，
测试本身就是要锁住跨引擎行为。若失败则按失败信息修 `combine`。

- [ ] **Step 3: 改 PROFILES 为引擎组合**

```python
PROFILES = {
    "lecture": {"engines": [{"name": "whisper", "chunk": 30}],
                "batch": 16, "beam": 5, "compute": "int8_float16"},
    "meeting": {"engines": [{"name": "whisper", "chunk": 30}, {"name": "funasr"}],
                "fallback_engines": [{"name": "whisper", "chunk": 30},
                                     {"name": "whisper", "chunk": 10}],
                "batch": 16, "beam": 5, "compute": "int8_float16"},
    "fast":    {"engines": [{"name": "whisper", "chunk": 30}],
                "batch": 16, "beam": 1, "compute": "int8_float16"},
}
```

`cli.py` 里按档位建引擎，FunASR 不可用时退回：

```python
def build_engines(cfg, compute, args):
    specs = cfg["engines"]
    try:
        return [(s, get_engine(s["name"], model_name=args.model,
                               device=args.device, compute=compute))
                for s in specs]
    except (ImportError, KeyError, OSError) as e:
        if "fallback_engines" not in cfg:
            raise
        log(f"⚠ 引擎不可用（{e}），退回 {cfg['fallback_engines']}")
        return [(s, get_engine(s["name"], model_name=args.model,
                               device=args.device, compute=compute))
                for s in cfg["fallback_engines"]]
```

每路的中间结果文件名按 `pass{i+1}.json` 保持不变，旧的 `.work/` 仍可复用。

- [ ] **Step 4: 跑全部测试**

Run: `python3 -m pytest tests/ -q`
Expected: 全绿

- [ ] **Step 5: 测出跨引擎的 CER 增量**

对同一份金标分别跑「whisper 双路」与「whisper + funasr」，记录两个 CER 到 `docs/measurements.md`。
**判定：跨引擎不优于同引擎双路时，`meeting` 档退回原配置**，并把结论写进 README。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "档位：支持引擎组合，跨引擎合并行为由测试锁住"
```

---

## Task 10: LLM 同音校订核心（可并行）

**Files:**
- Create: `audio_transcribe/polish.py`, `tests/test_polish.py`

**Interfaces:**
- Consumes: `audio_transcribe.evaluate.pinyin_key`
- Produces:
  - `constrain(before, after, loose=False) -> tuple[str, dict]` —— 纯函数，无网络
  - `polish(text, *, base_url, model, api_key, chunk=1200, dry_run=False) -> tuple[str, dict]`

**核心护栏**：LLM 只被允许做同音替换。返回文本长度必须与原文一致，否则整块拒绝；
逐字比对拼音，不一致的字一律还原为原字并计数。这样 LLM 无法删内容、无法增内容、
无法改事实——它能做的只有把 `旧港` 改成 `旧账`。

- [ ] **Step 1: 写失败测试**

```python
from audio_transcribe import polish as P


def test_homophone_edit_is_accepted():
    out, rep = P.constrain("旧港倒进去", "旧账倒进去")
    assert out == "旧账倒进去"
    assert rep["rejected"] == 0


def test_non_homophone_edit_is_reverted():
    """锁死：LLM 改了一个不同音的字，必须还原并记账。"""
    out, rep = P.constrain("旧港倒进去", "旧猫倒进去")
    assert out == "旧港倒进去"
    assert rep["rejected"] == 1


def test_length_change_rejects_whole_chunk():
    """长度变了说明 LLM 增删了内容——整块拒绝，绝不部分接受。"""
    out, rep = P.constrain("旧港倒进去", "旧账导进去了")
    assert out == "旧港倒进去"
    assert rep["reason"] == "length"


def test_deletion_is_rejected():
    out, rep = P.constrain("甲乙丙丁", "甲乙丁")
    assert out == "甲乙丙丁"
    assert rep["reason"] == "length"


def test_mixed_edits_keep_only_the_homophone_ones():
    out, rep = P.constrain("旧港和固定资产", "旧账和固定资全")
    assert out == "旧账和固定资产"
    assert rep["rejected"] == 1


def test_punctuation_change_is_reverted():
    out, _ = P.constrain("甲乙，丙丁", "甲乙。丙丁")
    assert out == "甲乙，丙丁"


def test_identical_text_is_noop():
    out, rep = P.constrain("完全一样", "完全一样")
    assert out == "完全一样" and rep["rejected"] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/test_polish.py -q`
Expected: FAIL，`No module named 'audio_transcribe.polish'`

- [ ] **Step 3: 实现 constrain**

```python
def constrain(before, after, loose=False):
    """只放行同音替换。长度不一致整块拒绝。"""
    if len(after) != len(before):
        return before, {"rejected": 0, "accepted": 0, "reason": "length"}
    out, rejected, accepted = [], 0, 0
    for a, b in zip(before, after):
        if a == b:
            out.append(a)
        elif pinyin_key(a, loose) == pinyin_key(b, loose) and pinyin_key(a, loose) != ():
            out.append(b); accepted += 1
        else:
            out.append(a); rejected += 1
    return "".join(out), {"rejected": rejected, "accepted": accepted, "reason": None}
```

`pinyin_key(a) != ()` 这个条件保证标点、字母、数字（拼音为空）不会被互换。

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest tests/test_polish.py -q`
Expected: `7 passed`

- [ ] **Step 5: 实现 polish 的网络部分**

```python
import json
import urllib.error
import urllib.request

PROMPT = (
    "你在校对中文语音识别稿。只允许把同音错别字改成正确的字。\n"
    "硬性要求：\n"
    "1. 输出字数必须与输入完全一致，一个字都不能增删\n"
    "2. 不得修改任何标点符号\n"
    "3. 只改读音相同的错别字，读音不同的一律不动\n"
    "4. 直接输出校对后的文本，不要解释、不要加引号\n"
)


def _chunks(text, size):
    """按句末标点切，避免把一句话劈成两块。"""
    out, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= size and ch in "。！？":
            out.append(cur); cur = ""
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
        except (urllib.error.URLError, KeyError, TimeoutError) as e:
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
```

用标准库 `urllib` 而不是 `requests`——少一个依赖，且 CI 不需要装它。

- [ ] **Step 6: Commit**

```bash
git add audio_transcribe/polish.py tests/test_polish.py
git commit -m "LLM 校订：拼音硬约束，只放行同音替换，长度变化整块拒绝"
```

---

## Task 11: 接入 polish 与 accurate 档

**Files:**
- Modify: `audio_transcribe/cli.py`

**Interfaces:**
- Produces: CLI 新增 `--polish` / `--llm-base-url` / `--llm-model` / `--polish-dry-run`；
  环境变量 `AUDIO_TRANSCRIBE_LLM_KEY` 读 key；`PROFILES` 新增 `accurate`

- [ ] **Step 1: 加 accurate 档与参数**

```python
"accurate": {"engines": [{"name": "whisper", "chunk": 30}, {"name": "funasr"}],
             "batch": 16, "beam": 5, "compute": "int8_float16", "polish": True},
```

`--polish` 默认关闭；key 只从环境变量读，**绝不进命令行参数**（会落进 shell history）。

- [ ] **Step 2: 接入顺序**

在 `combine` 之后、`render` 之前。polish 逐行处理以保住时间戳与 `words`：

```python
prep = None
if args.polish or cfg.get("polish"):
    from .polish import polish as run_polish
    key = os.environ.get("AUDIO_TRANSCRIBE_LLM_KEY", "")
    if not key and not args.polish_dry_run:
        raise SystemExit("需要环境变量 AUDIO_TRANSCRIBE_LLM_KEY")
    joined = "\n".join(r["text"] for r in rows)
    fixed, prep = run_polish(joined, base_url=args.llm_base_url,
                             model=args.llm_model, api_key=key,
                             dry_run=args.polish_dry_run)
    parts = fixed.split("\n")
    if len(parts) == len(rows):        # 行数对不上说明 LLM 破坏了结构，整体放弃
        for r, t in zip(rows, parts):
            r["text"] = t
    else:
        log(f"⚠ LLM 返回行数 {len(parts)} ≠ 原 {len(rows)}，本次校订整体作废")
        prep["structure_rejected"] = True
    # 兜底：再跑一次拼音纠错，覆盖 LLM 的任何反悔，代价接近零
    for r in rows:
        r["text"], _ = pinyin_fix(r["text"], terms)
    log(f"LLM 校订：{prep['chunks']} 块，接受 {prep['accepted']} 处、"
        f"拒绝 {prep['rejected']} 处、整块拒绝 {prep['length_rejected']} 块")
```

用换行拼接并校验行数，是因为 `constrain` 只保长度不保结构——多一道行数校验，
LLM 就无法把两行并成一行而不被发现。

- [ ] **Step 3: 质检报告与转录说明记账**

质检报告 json 加 `"polish": prep`（`prep` 为 `None` 表示未启用）。
`meta_lines` 里追加：

```python
if prep:
    meta_lines.append(
        f"**LLM 同音校订**　已启用，接受 {prep['accepted']} 处、拒绝 {prep['rejected']} 处"
        f"（拒绝的是读音不同的改动，一律还原为原字）。\n")
```

- [ ] **Step 4: dry-run 人工验证**

Run: `audio-transcribe run <音频> -o /tmp/t2 --profile accurate --polish-dry-run`
Expected: 打印将发送的块数与首块内容，不发任何网络请求

- [ ] **Step 5: 测出 polish 的 CER 增量**

对同一份金标跑「开 polish」与「关 polish」，记录到 `docs/measurements.md`。
**判定：CER 不改善则默认保持关闭**，并在 README 记录实测结论。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "接入 LLM 校订与 accurate 档，key 只走环境变量"
```

---

## Task 12: 速度基准与门禁

**Files:**
- Create: `bench/throughput.py`, `docs/measurements.md`

**Interfaces:**
- Produces: `audio-transcribe` 无关的独立脚本，打印 `实时倍数`

- [ ] **Step 1: 写 throughput.py**

```python
#!/usr/bin/env python3
"""默认档吞吐基准。

基线必须可复现，所以固定一段基准音频测吞吐，而不是拿「上次那份录音跑了多久」
当基线——那份录音的源音频已丢失，无法复跑。

    python3 bench/throughput.py --wav bench/data/bench15.wav
"""
import argparse
import statistics
import sys
import time

sys.path.insert(0, ".")
from audio_transcribe.audio import ensure_cuda_libs

BASELINE = 24.5         # bench15.wav（15 分钟）、int8_float16、batch16、beam5、词级时间戳开
TOLERANCE = 0.05        # 退化不得超过 5% → 下限 23.3x


def run_once(wav, words):
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    m = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    pipe = BatchedInferencePipeline(model=m)
    t0 = time.time()
    segs, info = pipe.transcribe(
        wav, language="zh", batch_size=16, beam_size=5, chunk_length=30,
        vad_filter=True, word_timestamps=words,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400),
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], no_speech_threshold=0.6)
    n = sum(len(s.text) for s in segs)      # 必须消费生成器，否则计时无意义
    el = time.time() - t0
    return info.duration / el, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    # 默认开——基线必须测流水线实际跑的配置
    ap.add_argument("--no-words", dest="words", action="store_false",
                    default=True, help="关词级时间戳（只用于对比，不是默认档配置）")
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    ensure_cuda_libs()

    run_once(a.wav, a.words)                # 预热，丢弃
    xs = [run_once(a.wav, a.words)[0] for _ in range(a.runs)]
    x = statistics.mean(xs)
    floor = BASELINE * (1 - TOLERANCE)
    print(f"实时倍数 {x:.2f}x（{a.runs} 次均值，各次 {[f'{v:.2f}' for v in xs]}）")
    print(f"基线 {BASELINE}x，下限 {floor:.2f}x → {'通过' if x >= floor else '不通过'}")
    return 0 if x >= floor else 1


if __name__ == "__main__":
    sys.exit(main())
```

`n = sum(...)` 那行不能删——faster-whisper 返回的是生成器，不消费就等于没转录，
计时会假到离谱。

- [ ] **Step 2: 测当前吞吐并与基线比对**

Run: `python3 bench/throughput.py --wav <15分钟基准片段>.wav`
Expected: 打印实时倍数。**与基线 24.5x 比，退化不得超过 5%（即不低于 23.3x）**

- [ ] **Step 3: 把所有实测数字汇总进 docs/measurements.md**

包含：各档吞吐、拼音纠错 CER 增量、跨引擎 CER 增量、polish CER 增量、
以及每项的默认开关判定与理由。

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "速度基准脚本与实测汇总"
```

---

## Task 13: 开源化（可与阶段二并行）

**Files:**
- Create: `LICENSE`, `.github/workflows/ci.yml`, `examples/terms/finance-lecture.json`
- Modify: `README.md`, `.gitignore`
- Delete: `terms/finance-lecture.json`（移动到 examples/ 并脱敏）

- [ ] **Step 1: MIT LICENSE**

标准 MIT 文本，版权行 `Copyright (c) 2026 xr843`。

- [ ] **Step 2: 术语表脱敏并移位**

```bash
mkdir -p examples/terms
git mv terms/finance-lecture.json examples/terms/finance-lecture.json
```

删除部分条目（如含具体单位名称、具体人名的映射），保留公开法规名与通用财税术语。
同时按新格式补 `terms` 字段（正确写法列表），供拼音纠错使用。

- [ ] **Step 3: GitHub Actions**

`.github/workflows/ci.yml`：

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: python -m pytest tests/ -q
```

只装核心依赖 + dev，不装 ASR 后端——CI 无 GPU，跑的是纯文本阶段与适配器纯函数。

- [ ] **Step 4: 验证 CI 装得动**

Run:
```bash
python3 -m venv /tmp/ci && /tmp/ci/bin/pip install -q -e ".[dev]" && /tmp/ci/bin/python -m pytest tests/ -q
```
Expected: 全绿。若有测试因缺 `faster_whisper` 失败，说明该测试在模块顶层 import 了 ASR 后端——
**改测试让 import 延迟，不要把 ASR 后端加进 CI 依赖。**

- [ ] **Step 5: README 补开源必需内容**

在文件最顶部（标题下方）插入英文摘要：

```markdown
> **English summary.** A long-audio Chinese transcription pipeline built for
> *completeness*, not for a single pass. Running Whisper once always drops
> content — and worse, you never find out where. This pipeline cross-runs two
> ASR engines, audits the timeline for gaps by measuring actual RMS loudness,
> detects content swallowed *inside* segments by character density, re-transcribes
> only the problem spans, and reports character-level CER against a gold set.
> Runs fully offline on a local GPU; audio never leaves the machine.
> The "踩过的坑" section below is the real value here — 16 findings, every one
> measured on real recordings rather than reasoned from documentation.
>
> `pip install -e ".[whisper]"` then `audio-transcribe run recording.mp3`.
> Docs are in Chinese; the code and CLI are not.
```

再改三处：

- 「环境」小节的安装命令改为 `pip install --user --break-system-packages -e ".[whisper]"`，
  并说明 `[funasr]` 为可选第二引擎
- 术语表路径全部由 `terms/` 改为 `examples/terms/`
- 新增「## 正确率评测」小节，写清三步流程：

```markdown
## 正确率评测

流水线过去只报覆盖率——转到了多少；不报正确率——转对了多少。覆盖率 100% 的稿子
可以句句是错的。三步建立自己的尺子：

```bash
# 1. 从跑完的输出里切一段生成待校对稿（初稿就是 ASR 结果）
audio-transcribe goldset 输出目录/ --from 00:10:00 --to 00:25:00 -o sample.gold.tsv

# 2. 打开 sample.gold.tsv，只改错字。不碰格式、不打时间戳。

# 3. 评测
audio-transcribe eval --gold sample.gold.tsv --hyp 输出目录/
```

报告里 **删除率** 是「不遗漏」的直接度量；**同音/近音替换率** 告诉你拼音纠错与
LLM 校订的天花板在哪里。各档实测数字见 [docs/measurements.md](docs/measurements.md)。
```

- [ ] **Step 6: 跑全部测试并提交**

Run: `python3 -m pytest tests/ -q`
Expected: 全绿

```bash
git add -A
git commit -m "开源化：MIT 许可、GitHub Actions、术语表脱敏移入 examples/"
```

---

## 完成判定

对照 spec 的验收标准逐条勾：

- [ ] `eval` 在 CI 夹具上跑出确定的 CER，可重复
- [ ] `goldset` → 人工改错字 → `eval` 全链路走通
- [ ] 拿到一份音频仍在的真实录音的 CER 与同音/近音替换率
- [ ] 拼音纠错报出独立 CER 增量；不劣化才默认开启
- [ ] FunASR 作为第二路跑通；跨引擎合并有测试证明不丢内容
- [ ] `--polish` 的拼音拒绝率被记账；单测证明不同音改动 100% 被拒
- [ ] 默认档速度退化 ≤ 5%（≥ 23.3x 实时）
- [ ] `pip install -e .` 后 `audio-transcribe` 可用
- [ ] GitHub Actions 绿
- [ ] 全部现有 24 个测试仍然通过
