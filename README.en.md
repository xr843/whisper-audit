# WhisperAudit

**[English](https://github.com/xr843/whisper-audit/blob/master/README.en.md) · [中文](https://github.com/xr843/whisper-audit/blob/master/README.md)**

[![PyPI](https://img.shields.io/pypi/v/whisper-audit)](https://pypi.org/project/whisper-audit/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xr843/whisper-audit/blob/master/examples/colab_demo.ipynb)
[![CI](https://github.com/xr843/whisper-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/xr843/whisper-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/xr843/whisper-audit/blob/master/LICENSE)

A transcription pipeline for long-form Chinese audio that produces transcripts,
subtitles, and a QA report. The design goal is **completeness**: beyond
transcribing once, it audits its own output — locating spans that contain
speech but produced no text by measuring actual loudness, re-transcribing only
the problem spans, and shipping a built-in character-level CER harness so
accuracy can be verified on your own recordings.

**Try it online**: the Colab badge above transcribes a real recording in your
browser (free T4 GPU, measured at 30.8× realtime) with nothing installed.

## Features

- **Coverage auditing** — loudness-based detection of dropped speech, including
  drops hidden *inside* long segments
- **Targeted re-transcription** — only flagged spans are re-run, never the whole file
- **Three selectable engines** — faster-whisper, FunASR Paraformer, Qwen3-ASR,
  with published measurements across six public domains (below)
- **Speaker diarization** (optional) — voiceprint clustering labels `S1`/`S2`;
  labels are added, text is never modified
- **Glossary correction** — literal replacement plus pinyin fuzzy matching; every
  automated edit is logged with a timestamp for review
- **Built-in evaluation** — `goldset` / `eval` subcommands produce CER with error
  breakdown on your own recordings
- **QA report** — coverage, removed hallucinations, remaining suspicious spans,
  and an edit ledger, produced with every run
- **Fully offline** — no network traffic for audio or text by default (the only
  exception, `--polish`, is off by default)
- **Three interfaces** — CLI, local web UI, and Colab

## Measured results (2026-08-07)

Six domains × three engines, one scoring protocol (character error rate, lower
is better). First three columns are clean speech, last three are hard:

| Engine | Talk | Lecture | Read | Accent+fast | Crosstalk | Livestream |
|---|---|---|---|---|---|---|
| `--engine funasr` | **2.06** | **2.44** | 5.36 | 24.88 | 32.76 | 25.69 |
| `--engine qwen` | 2.11 | 2.61 | **3.92** | 33.99 | 15.05 | 27.10 |
| default whisper | 4.18 | 8.67 | 4.45 | **24.63** | **11.86** | **14.78** |

Models in the table: whisper is faster-whisper large-v3, funasr is
SeacoParaformer large, qwen is Qwen3-ASR-0.6B.
Corpora: five SpeechIO test sets plus FLEURS (commercial APIs score roughly
1.5–3% on the clean SpeechIO sets). Summary: on clean Mandarin `funasr` is the
most accurate with 6–10× fewer deletions; on hard audio whisper leads in all
three domains while the other two each degrade sharply. **The default engine is
therefore whisper — the only one that never collapsed — rather than the one
with the best average.**

Ablation on identical audio and reference shows the audit and repatch add
nothing on clean Mandarin, and the dual-pass merge (formerly the default) made
CER 2.3× worse — the default profile is now single-pass. On hard audio the dual
pass brings CER from 88.6% down to 74.2%, so `--profile meeting` is retained
for audio with confirmed dropped content. All data, including results
unfavorable to this project and conclusions that were later overturned, is in
[docs/measurements.md](https://github.com/xr843/whisper-audit/blob/master/docs/measurements.md).

| Speed | Measured |
|---|---|
| Long audio (GPU) | 24.5× default, 62× with `--profile fast` (RTX 4060 Laptop 8GB); 30.8× on Colab T4 |
| CPU-only | 3.2× realtime (`--engine funasr --device cpu`), no discrete GPU required |

## Requirements

| Item | Requirement |
|---|---|
| Python | ≥ 3.10 |
| ffmpeg | Required (audio transcoding); install via the system package manager |
| OS | Linux and WSL2 tested; macOS expected to work (CPU path); native Windows untested |
| GPU | Optional. NVIDIA (tested with 8GB VRAM); a CPU-only path is available |
| Disk | Roughly 1–3GB of models per engine, downloaded on first run |

## Installation

```bash
pip install "whisper-audit[whisper,cuda]"   # GPU; omit cuda for CPU-only
sudo apt install ffmpeg                     # macOS: brew install ffmpeg
```

Optional dependencies are grouped by engine and feature, and can be combined:

| Extra | Contents |
|---|---|
| `whisper` | faster-whisper engine (the default engine) |
| `cuda` | CUDA 12 runtime libraries. Needed only for the whisper engine on GPU; environments that already have torch do not need it |
| `funasr` | FunASR Paraformer engine |
| `qwen` | Qwen3-ASR engine. Note: its dependencies pin `transformers` to 4.57.6 (downgrading an existing install); a separate virtual environment is recommended |
| `ui` | Local web interface |
| `socks` | Required for model downloads behind a SOCKS proxy |

Installation notes (all from clean-environment testing):

- A GPU environment with only `[whisper]` and no CUDA runtime fails at
  transcription start with `Library libcublas.so.12 is not found`; installing
  `[cuda]` resolves it
- Model downloads behind a SOCKS proxy without `socksio` fail with
  `Using SOCKS proxy, but the 'socksio' package is not installed`; installing
  `[socks]` resolves it

Common formats (mp3 / m4a / wav, and anything ffmpeg reads) are supported;
audio is normalized to 16kHz mono.

## Usage

```bash
# Standard Mandarin (talks / lectures / meetings): most accurate and fastest in this domain
#   the engine needs its extra: pip install "whisper-audit[funasr]"
whisper-audit run recording.mp3 --engine funasr

# General case (default configuration): single pass + coverage audit + targeted repatch
whisper-audit run recording.mp3

# Hard audio with confirmed dropped content (heavy noise / sustained vocals): dual-pass
whisper-audit run recording.mp3 --profile meeting

# Speed-first: turbo model at 62× realtime, measured quality cost ~0.9pp
whisper-audit run recording.mp3 --profile fast

# Multiple speakers: label speakers (labels only; text is never modified)
whisper-audit run interview.mp3 --engine funasr --diarize

# Local web interface: upload, monitor, and download in a browser
#   install: pip install "whisper-audit[whisper,ui]"
whisper-audit ui
```

The web interface binds to `127.0.0.1` by default. Started with
`whisper-audit ui --listen 0.0.0.0`, other devices on the local network can use
this machine's transcription service from a browser with nothing installed on
the client; in that mode audio travels over the local network to the serving
machine — enable it according to your environment's security requirements.

![whisper-audit ui: upload and progress on the left, QA summary and transcript preview on the right](https://raw.githubusercontent.com/xr843/whisper-audit/master/docs/images/webui.jpg)

(Real run shown: the same *True Story of Ah Q* reading as the Colab demo, at
26.1× realtime on this run.)

Running from source: `python3 transcribe.py recording.mp3` is equivalent to the
installed command. Remaining flags (`--terms`, `--keep-break`, `--speakers`,
`--model`, `--device`, `--language`, `--title`, `-o`) are described in `--help`.

### Choosing an engine

- Clear Mandarin → `funasr`
- Well-recorded mixed material, **short** files → `qwen` (measured at only
  0.4–1.2× realtime on long audio; not suitable for long recordings)
- Heavy accent, noise, amateur or far-field recordings, or unknown material →
  the whisper default. A quiet, reverberant amateur reading degraded funasr to
  35% CER while whisper held 13% on the same file

### Output

Written to `<name>_转录/` next to the audio:

| File | Contents |
|---|---|
| `*_全文转录.md` / `.txt` | Transcript with timestamped paragraphs and automatic punctuation; with `--diarize`, paragraphs break on speaker change and carry `S1`/`S2` |
| `*_字幕.srt` | Subtitles cut on word-level timestamps for verification against the audio; with `--diarize`, every cue carries a speaker label |
| `质检报告.json` | QA report: coverage, dropped spans, hallucinations, edit ledger; reviewing it before delivering a transcript is recommended |
| `.work/` | Intermediate results, reused on re-runs (delete before changing parameters) |

The final log line reports the delivered transcript's actual coverage and the
number of remaining suspicious spans. A real run on the LibriVox public-domain
reading of *The True Story of Ah Q*, chapter 1 (7.8 minutes — the same audio as
the Colab demo):

```
审计：覆盖 95.7%　讲话中位音量 -27.2dB　待补 0 处（其中段内饥饿 0）　幻觉 0 处
终审：合并稿覆盖 95.7%　有效语音 95.7%　残余饥饿段 0 处
完成：6 段 / 1,646 字 / 64 条字幕
```

(Log labels are Chinese, like the output filenames. Line by line: *audit* —
coverage 95.7%, median speech level −27.2dB, 0 spans to repatch of which 0
starved, 0 hallucinations; *final review* — merged-transcript coverage 95.7%,
effective speech 95.7%, 0 starved spans remaining; *done* — 6 paragraphs /
1,646 characters / 64 subtitle cues.)

The only uncovered span is the last 5.1 seconds — the QA report pinpoints its
exact interval, and loudness measurement shows no speech in it, so it is not
re-transcribed.

## Measuring accuracy on your own recordings

```bash
# 1. Cut a window of the output into a correction sheet (seeded with the ASR text)
whisper-audit goldset output_dir/ --from 00:10:00 --to 00:20:00 -o sample.gold.tsv
# 2. Edit sample.gold.tsv: fix only wrong characters in column 3; leave timestamps and rows intact
# 3. Score
whisper-audit eval --gold sample.gold.tsv --hyp output_dir/
```

The report includes CER, the **deletion rate** (the project's primary metric,
measuring dropped content), and the **homophone error share**, which bounds
what any pinyin-based correction can achieve. `eval --manifest` scores public
benchmark sets directly.

## Glossary (optional)

```json
{ "name": "domain",
  "fixes": [["misrecognized form", "correct form"]],
  "terms": ["correct form"] }
```

- `fixes` — exact literal replacement; reviewing entries individually is
  recommended. The tool reports per-entry hit counts, exposing entries that
  never fire or fire suspiciously often
- `terms` — pinyin fuzzy matching; one entry covers a class of homophone
  errors. Off by default; enabled explicitly with `--pinyin-fix`

Example: `examples/terms/finance-lecture.json` (42 fixes + 41 terms, verified
against real recordings).

## Optional features that modify text (all off by default)

| Flag | Effect | Why it is off by default |
|---|---|---|
| `--pinyin-fix` | Pinyin-level glossary correction | Toneless pinyin conflates genuinely different words (节余 *surplus* / 结余 *balance*) |
| `--polish` | LLM homophone repair | The transcript is sent to the configured endpoint, and homophone substitution can itself change meaning |

When enabled, every edit is logged to the QA report with a timestamp and should
be reviewed. `--polish` takes `--llm-base-url` / `--llm-model` /
`--polish-dry-run`; the API key is read only from the `WHISPER_AUDIT_LLM_KEY`
environment variable. Its guardrail enforces that every replacement is a true
homophone and that the line count is unchanged, so content cannot be added or
removed — but a homophone swap can still change meaning, which is why it
remains opt-in.

## Known limitations

1. ASR accuracy has a ceiling: strong accents and dense terminology produce
   clustered errors. The glossary fixes recurring ones; the rest requires human
   review against the `.srt`
2. Punctuation is inferred from inter-word pauses and connectives; it does not
   represent the speaker's intended phrasing
3. Speaker labels are inferred by voiceprint clustering: short interjections may
   be attributed to the wrong speaker, and no labels are added when only one
   speaker is detected. Diarization never modifies text (locked by a test)
4. Singing is out of scope: slow sustained vocals measured 74%+ CER,
   deletion-dominated, and re-transcription does not recover it
5. Automatic break detection can misfire; `--keep-break` disables it

## Documentation

- [docs/lessons.md](https://github.com/xr843/whisper-audit/blob/master/docs/lessons.md) — 24 field-tested engineering notes: VAD
  deleting real speech, silence hallucinations, intra-segment drops, merge
  pitfalls, contaminated performance measurements — each from an actual
  measurement
- [docs/measurements.md](https://github.com/xr843/whisper-audit/blob/master/docs/measurements.md) — every number and the
  justification for every default, with overturned conclusions kept verbatim
- [Releases](https://github.com/xr843/whisper-audit/releases) — version history

Documentation under `docs/` is in Chinese; the code, CLI, and this README are
in English.

## Contributing

```bash
pip install -e ".[dev]"        # pure-CPU core dependencies, no ASR backend
python3 -m pytest tests/ -q    # completes in seconds, no GPU or audio required
```

The test suite deliberately runs without a GPU or audio files: engine adapters
are pure functions over recorded real-world outputs, and diarization tests use
real voiceprint embeddings saved as fixtures. The tests target the class of
defect that never raises an exception — silently dropping content, silently
rewriting text, silently overstating coverage.

Setup, testing discipline, and where help is most needed:
[CONTRIBUTING.md](https://github.com/xr843/whisper-audit/blob/master/CONTRIBUTING.md).

```
whisper_audit/
  cli.py         CLI and main flow        engines/   ASR backends (whisper / funasr / qwen)
  audio.py       transcode + loudness     audit.py   gaps / hallucinations / starvation
  merge.py       multi-pass merge         render.py  punctuation / paragraphs / subtitles
  evaluate.py    character-level CER      goldset.py correction sheets and scoring
  terms.py       glossary: literal+pinyin polish.py  LLM homophone repair (pinyin-locked)
  diarize.py     speaker embeddings       webui.py   local web interface
```

## Acknowledgements

Built on the following open-source work:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / CTranslate2 and the OpenAI Whisper models
- [FunASR](https://github.com/modelscope/FunASR) (SeacoParaformer and the cam++ speaker model)
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Gradio](https://github.com/gradio-app/gradio) (web interface)
- Evaluation corpora: [SpeechIO Leaderboard](https://github.com/SpeechColab/Leaderboard),
  [FLEURS](https://huggingface.co/datasets/google/fleurs);
  demo audio from [LibriVox](https://librivox.org/) (public domain)

## License

[MIT](https://github.com/xr843/whisper-audit/blob/master/LICENSE)
