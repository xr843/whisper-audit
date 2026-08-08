# WhisperAudit

**[English](README.en.md) · [中文](README.md)**

[![PyPI](https://img.shields.io/pypi/v/whisper-audit)](https://pypi.org/project/whisper-audit/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xr843/whisper-audit/blob/master/examples/colab_demo.ipynb)
[![CI](https://github.com/xr843/whisper-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/xr843/whisper-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Try without installing**: hit the Colab badge above — a real recording transcribed in your browser in ~5 minutes, measured at 30.8× realtime on the free T4.

Long-form Chinese audio → transcript and subtitles. The goal is **leaving nothing
out**, not "running it through once": the pipeline audits its own output for
dropped speech, re-transcribes only the problem spans, optionally labels
speakers, and ships a reproducible accuracy ruler so you can measure the result
on *your* recordings instead of trusting a README.

Three selectable engines. Runs offline on a local GPU or **CPU-only**;
audio never leaves your machine.

## Measured results (2026-08-07)

Six domains × three engines, one ruler (character error rate, lower is better).
First three columns are clean domains, last three are hard:

| Engine | Talk | Lecture | Read | Accent+fast | Crosstalk | Livestream |
|---|---|---|---|---|---|---|
| `--engine funasr` | **2.06** | **2.44** | 5.36 | 24.88 | 32.76 | 25.69 |
| `--engine qwen` | 2.11 | 2.61 | **3.92** | 33.99 | 15.05 | 27.10 |
| default whisper | 4.18 | 8.67 | 4.45 | **24.63** | **11.86** | **14.78** |

Corpora: five SpeechIO sets + FLEURS (commercial APIs score ~1.5–3% on the clean
SpeechIO sets). The pattern in one sentence: **on clean Mandarin `funasr` is the
most accurate and drops 6–10× fewer characters; on hard audio whisper wins every
domain while the other two each fall off a cliff — the default slot goes to the
most predictable engine, not the best average.**

**What does the pipeline itself buy?** Ablation on the same audio and reference:
on clean Mandarin the audit + repatch buy **exactly nothing**, and the dual-pass
merge (the old default) made CER 2.3× worse — the default is single-pass now.
On hard audio the dual pass pulls 88.6% back to 74.2%, so `--profile meeting`
stays as a **recovery option for audio you have confirmed is losing content**.
The unflattering numbers are all in [docs/measurements.md](docs/measurements.md),
overturned conclusions kept verbatim.

| Speed | Measured |
|---|---|
| Long audio (GPU) | 24.5× default / 62× `--profile fast` (RTX 4060 8GB); 30.8× on a free Colab T4 |
| CPU-only | 3.2× (`--engine funasr --device cpu`) — an ordinary laptop is enough |

## What makes this different

Most transcription wrappers hand you text and let you assume it's complete.
This one assumes it isn't, and checks:

- **Coverage audit from actual loudness.** The pipeline measures dBFS across the
  timeline and flags spans that sound like speech but produced no text. Silence
  and dropped speech are different problems and get told apart.
- **Intra-segment starvation detection.** VAD trims silence and maps timestamps
  back to the original axis, so one segment can span 296 seconds while
  containing 128 characters — and naive coverage math counts all 296 seconds as
  "covered". This is the single biggest blind spot in gap-based auditing, and it
  is checked explicitly.
- **Targeted re-transcription.** Only flagged spans get a second pass, with
  different chunking. Cross-engine by design: the repatch pass uses whisper even
  when the main pass didn't.
- **A QA report you're meant to read.** Coverage, dropped spans, hallucination
  list, and a line-by-line ledger of every automated edit with timestamps.
- **Your own CER, on your own audio.** `goldset` cuts a window of the output into
  a correction sheet; you fix only the wrong characters; `eval` scores it and
  reports CER split into substitutions / deletions / insertions, plus what
  fraction of the errors are homophones (which caps what any pinyin-based or
  LLM-based repair can ever fix).

## Install

```bash
git clone https://github.com/xr843/whisper-audit && cd whisper-audit
pip install -e ".[whisper,cuda]"  # GPU; drop `cuda` for CPU-only
sudo apt install ffmpeg           # system dependency; macOS: brew install ffmpeg
```

Engines are extras and combinable: `whisper` / `funasr` / `qwen`.

**Two things a clean machine will hit** (found by cold-start testing on
2026-08-07, now encoded in the extras):

- **`cuda` is only needed for the whisper engine on GPU.** CTranslate2 does not
  bundle the CUDA runtime, so `[whisper]` alone fails with
  `Library libcublas.so.12 is not found` once transcription actually starts.
  It is a separate extra because those two wheels are ~700MB. `funasr` and
  `qwen` go through torch, which ships those libraries itself — they don't need
  `cuda`, and neither does any environment that already has torch.
- **`qwen` is a heavy install.** `qwen-asr` pins `transformers==4.57.6` exactly
  (so it will downgrade yours) and hard-depends on gradio and flask. Give it its
  own venv if anything else in your environment needs a newer transformers.
- **Behind a SOCKS proxy, add `socks`.** `huggingface_hub` moved to httpx, which
  needs `socksio` to download through a SOCKS proxy.

Models download on first run (large-v3 is ~2.9GB). mp3 / m4a / wav and other
common formats work — ffmpeg normalizes everything to 16kHz mono.

**No GPU?** `--engine funasr --device cpu` runs at 3.2× realtime. Paraformer is
non-autoregressive, so its CPU inference is far faster than whisper's.

## Usage

```bash
# Clear Mandarin (talks, lectures, meetings) — best quality and fast
whisper-audit run recording.mp3 --engine funasr

# Clean mixed material, SHORT files — best on read speech; avoid for long audio (0.4–1.2× realtime)
whisper-audit run short_clip.mp3 --engine qwen

# Not sure — default: single pass + coverage audit + targeted repatch
whisper-audit run recording.mp3

# You've confirmed a single pass is dropping content (noise, sustained vocals) — dual-pass
whisper-audit run recording.mp3 --profile meeting

# In a hurry — turbo model, 62× realtime, measured quality cost only 0.9pp
whisper-audit run recording.mp3 --profile fast

# Multi-speaker — label speakers (labels only; never edits a character of text)
whisper-audit run interview.mp3 --engine funasr --diarize
```

`python3 transcribe.py recording.mp3` is equivalent to the installed command.

Other flags: `--terms glossary.json`, `--keep-break`, `--speakers N`,
`--model`, `--device`, `--language`, `--title`, `-o`. See `--help`.

### Choosing an engine

Clear Mandarin → `funasr`. Clean mixed material, **short** files → `qwen`.
Heavy accent / noise / not sure → the whisper default — the only engine that
never collapsed across all six measured domains (table above).

### Output

Written to `<name>_转录/` next to the audio:

| File | Purpose |
|---|---|
| `*_全文转录.md` / `.txt` | Transcript, timestamped paragraphs, auto punctuation. With `--diarize`, paragraphs break on speaker change and carry `S1`/`S2` |
| `*_字幕.srt` | Subtitles cut on word-level timestamps, for listening back against the audio. With `--diarize`, every cue carries `[S1]` |
| `质检报告.json` | QA report — coverage, dropped spans, hallucinations, edit ledger. **Read this before you ship the transcript** |
| `.work/` | Intermediate results, reused on re-runs (delete before changing parameters) |

The final log line reports the merged transcript's real coverage and any
remaining suspicious spans.

## Measure accuracy on your own recording

```bash
# 1. Cut a window of the output into a correction sheet (seeded with the ASR text)
whisper-audit goldset output_dir/ --from 00:10:00 --to 00:20:00 -o sample.gold.tsv
# 2. Open it and fix ONLY wrong characters in column 3 — don't touch timestamps, don't merge rows
# 3. Score it
whisper-audit eval --gold sample.gold.tsv --hyp output_dir/
```

You get CER, the **deletion rate** (how much went missing — this project's
headline metric), and the **homophone error share**, which tells you the ceiling
of any pinyin- or LLM-based correction before you bother enabling one.

You can also score public benchmarks directly:
`whisper-audit eval --manifest fleurs.jsonl --engine qwen`.

## Glossary (optional)

```json
{ "name": "domain",
  "fixes": [["misrecognized form", "correct form"]],
  "terms": ["correct form"] }
```

- `fixes` — exact literal replacement, reviewed one by one. **Count occurrences
  before adding an entry**; the tool reports hit counts so you can spot entries
  that fire zero times or suspiciously often.
- `terms` — pinyin fuzzy matching, one entry covers a class of homophone errors.
  **Off by default**; enable with `--pinyin-fix`.

## Two switches that rewrite your text — both off by default

| Flag | What it does | Why it's off |
|---|---|---|
| `--pinyin-fix` | Pinyin-level glossary correction | Toneless pinyin merges genuinely different words (节余 *surplus* / 结余 *balance*). No glossary can fix that class |
| `--polish` | LLM homophone repair | **Sends your transcript to the endpoint you configure**; and homophone substitution can itself change meaning |

Both log every edit with a timestamp into the QA report, and you are expected to
review them. `--polish` reads its key only from `WHISPER_AUDIT_LLM_KEY`
(never a command-line flag — that lands in shell history) and supports
`--polish-dry-run`. Its guardrail enforces that every replacement is a true
homophone of the original *and* that the line count is unchanged, so it cannot
add or remove content — but a homophone swap can still change meaning
(权力 *power* / 权利 *right*), which is why it stays opt-in.

## Known limitations

1. **ASR accuracy has a ceiling.** Dense terminology and strong accents produce
   clustered errors. The glossary fixes recurring ones; the rest needs a human
   listening against the `.srt`.
2. **Punctuation is inferred** from inter-word pauses and connectives. It does
   not reflect the speaker's intended phrasing.
3. **Speaker labels are optional and inferred** (`--diarize`, off by default).
   Short interjections are easily attributed to the wrong person, and nothing is
   labeled when only one speaker is found. Diarization **never edits the text** —
   a test locks that invariant.
4. **Singing doesn't work.** Slow sustained vocals measured 74%+ CER, deletion-
   dominated; re-transcription does not recover it.
5. Automatic break detection can misfire; `--keep-break` disables it.

## Further reading

- **[docs/lessons.md](docs/lessons.md)** — 24 field-tested traps: VAD killing real
  speech, silence hallucinations, intra-segment starvation, merge pitfalls,
  performance measurements that turned out to be contaminated. Each one came
  from a measurement, not from reading documentation. **This file is the most
  valuable thing in the repo.**
- **[docs/measurements.md](docs/measurements.md)** — every number, and the
  justification for every default, including conclusions that were later
  overturned (kept verbatim, so you can see which way the evidence moved).

Both are in Chinese; the code, CLI and this README are not.

## Contributing

PRs welcome. Setup, testing discipline, and where help is most needed:
[CONTRIBUTING.md](CONTRIBUTING.md).


```bash
pip install -e ".[dev]"        # pure-CPU core deps, no ASR backend
python3 -m pytest tests/ -q    # 218 tests + 1 xfail, seconds, no GPU or audio needed
```

The test suite deliberately runs without a GPU or any audio file: engine
adapters are pure functions over recorded real-world outputs, and diarization
tests run against real cam++ embeddings saved as a fixture. What the tests
protect is the class of bug that never raises an exception — silently dropping
content, silently rewriting text, silently overstating coverage.

```
whisper_audit/
  cli.py         CLI and main flow        engines/   ASR backends (whisper / funasr / qwen)
  audio.py       transcode + loudness     audit.py   gaps / hallucinations / starvation
  merge.py       multi-pass merge         render.py  punctuation / paragraphs / subtitles
  evaluate.py    character-level CER      goldset.py correction sheets and scoring
  terms.py       glossary: literal+pinyin polish.py  LLM homophone repair (pinyin-locked)
  diarize.py     speaker embedding + clustering
```

MIT License.
