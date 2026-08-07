# audio-transcribe

> **English summary.** A long-audio Chinese transcription pipeline built for
> *completeness*: it audits the timeline for coverage gaps by measuring actual
> loudness, detects speech swallowed *inside* segments, re-transcribes only the
> problem spans, and ships its own character-level CER evaluation harness.
> Two engines — whisper large-v3 (robust default) and FunASR Paraformer
> (**2.06–2.44% CER on SpeechIO sets**, commercial-API territory). Runs offline
> on a local GPU or **CPU-only (3.2× realtime)**; audio never leaves the machine.
> Every default is backed by a measured number; `docs/` keeps the receipts.
>
> `pip install -e ".[whisper]"` then `audio-transcribe run recording.mp3`.
> Docs are in Chinese; the code and CLI are not.

中文长音频 → 文稿/字幕。目标是**不遗漏**，不是「转一遍」：
自动审计漏转、定点补转、并给你一把可复现的正确率尺子。
双引擎，本机 GPU 或**纯 CPU** 离线运行，音频不出本机。

## 实测成绩（2026-08-06）

| 语料 | 域 | 本流水线 | 参照 |
|---|---|---|---|
| SpeechIO ZH00004（场馆演讲） | 自发语音 | **2.06%**（`--engine funasr`） | 商用 API 同集约 1.5~3% |
| SpeechIO ZH00005（在线讲课） | 自发语音 | **2.44%**（`--engine funasr`） | 同上 |
| FLEURS cmn_hans test 全量 | 标准朗读 | 4.45%（whisper 默认档） | 官方报告 whisper 同集约 4.1% |
| 长音频吞吐 | 生产工况 | 默认档 24.5x / fast 档 62x 实时 | RTX 4060 Laptop 8GB |
| 纯 CPU 吞吐 | 无显卡场景 | 3.2x 实时（`--engine funasr --device cpu`） | 普通笔记本即可 |

字级 CER。口径、复现步骤与所有开关的判定依据见 [docs/measurements.md](docs/measurements.md)。

## 安装

```bash
git clone https://github.com/xr843/audio-transcribe && cd audio-transcribe
pip install -e ".[whisper]"          # whisper 引擎；两个都要装 ".[whisper,funasr]"
sudo apt install ffmpeg              # 系统依赖，pip 装不了；macOS: brew install ffmpeg
```

模型首次运行自动下载（large-v3 约 2.9GB）。mp3 / m4a / wav 等常见格式都支持
（ffmpeg 统一转 16kHz 单声道）。

## 用法

```bash
# 标准普通话（演讲/讲课/会议）—— 质量最高且快：CER 2% 级、删除少 6~10 倍
audio-transcribe run 录音.mp3 --engine funasr

# 口音重 / 内容复杂 / 拿不准 —— 默认档：whisper 双路交叉 + 审计补转，最全
audio-transcribe run 录音.mp3

# 单人讲授、音质好 —— 单路，快一半
audio-transcribe run 录音.mp3 --profile lecture

# 赶时间 —— turbo 引擎 62x 实时，质量代价实测仅 0.9pp
audio-transcribe run 录音.mp3 --profile fast
```

`python3 transcribe.py 录音.mp3` 与装包后的命令等价。
无显卡：`--engine funasr --device cpu`（3.2x 实时）。
其他参数：`--terms 术语表.json`、`--keep-break`（不剔除中场休息）、
`--model`、`--device`、`--language`、`--title`、`-o`，详见 `--help`。

**引擎一句话**：清晰普通话 → `funasr`；口音重、类型未知、歌唱类 → 默认 whisper
（funasr 在慢速歌唱域会崩，方言重口音未实测）。

### 输出

默认写到音频同目录的 `<名称>_转录/`：

| 文件 | 用途 |
|---|---|
| `*_全文转录.md` / `.txt` | 正文，时间戳分段、自动标点 |
| `*_字幕.srt` | 词级时间戳切分，配原音频逐句回听核对 |
| `质检报告.json` | 覆盖率、漏转区段、幻觉清单、改动记账——**出稿前看一眼** |
| `.work/` | 中间结果，重跑自动复用（改参数前先删） |

日志末尾的「终审」行报告合并稿的实际覆盖率与残余可疑段。

## 测一测你这份录音转得多准

```bash
# 1. 从输出切一段生成待校对稿（初稿就是 ASR 结果）
audio-transcribe goldset 输出目录/ --from 00:10:00 --to 00:20:00 -o sample.gold.tsv
# 2. 打开 sample.gold.tsv，只改第三列的错字——不动时间戳、不合并行
# 3. 评测
audio-transcribe eval --gold sample.gold.tsv --hyp 输出目录/
```

报告含 CER、**删除率**（漏了多少——本工具的立身指标）与**同音/近音错占比**
（决定拼音类修正手段的天花板）。

## 术语表（可选）

```json
{ "name": "领域名",
  "fixes": [["误识写法", "正确写法"]],
  "terms": ["正确写法"] }
```

- `fixes`：字面精确替换，人工逐条确认，**加条目前先统计频次**防误伤
- `terms`：拼音模糊匹配，一条覆盖一类同音错——**默认关闭**，`--pinyin-fix` 显式开

示例：`examples/terms/finance-lecture.json`（42 fixes + 41 terms，真实录音实测核定）。

## 两个会自动改正文的开关，默认都关

| 参数 | 作用 | 默认关的原因 |
|---|---|---|
| `--pinyin-fix` | 拼音级术语纠错 | 无调拼音会把「节余/结余」这类真实不同的词判成同一个 |
| `--polish` | LLM 同音校订 | **正文会发往你配置的 endpoint**；且同音替换本身可能改变含义 |

开启后必须逐条核对 `质检报告.json` 里的改动清单（每条带时间戳可回听）。
`--polish` 配套 `--llm-base-url` / `--llm-model` / `--polish-dry-run`，
key 只从环境变量 `AUDIO_TRANSCRIBE_LLM_KEY` 读。

## 已知局限

1. **ASR 准确率有上限**：方言口音 + 术语密集段落错得密，术语表修高频错，
   其余需对照 `.srt` 回听校对
2. **标点是推定的**（词间停顿 + 连接词），不代表讲者原意
3. **不做说话人分离**
4. **歌唱类音频不适用**（慢速拖腔实测 CER 74%+，删除主导，补转救不回）
5. 中场休息自动识别可能误判，`--keep-break` 可关

## 深入阅读

- **[docs/lessons.md](docs/lessons.md)** —— 16 条实测踩坑记录：VAD 误杀、静音幻觉、
  段内饥饿、合并陷阱、性能真相……每一条都是测出来的，不是文档推理。
  **这份记录是本项目真正的价值所在。**
- **[docs/measurements.md](docs/measurements.md)** —— 全部实测数字与每个默认值的
  判定依据，包括被推翻过的结论（原样保留）

## 开发

```bash
pip install -e ".[dev]"       # 纯 CPU 核心依赖，不装 ASR 后端
python3 -m pytest tests/ -q   # 148 tests + 1 xfail，秒级，无需 GPU/音频
```

```
audio_transcribe/
  cli.py         命令行与主流程          engines/   ASR 后端（whisper / funasr）
  audio.py       转码 + 音量测量          audit.py   空洞 / 幻觉 / 段内饥饿 / 终审
  merge.py       多路合并与去重           render.py  标点 / 分段 / 出稿 / 字幕重切
  evaluate.py    字级 CER 与增删改分解     goldset.py 待校对稿生成与评测入口
  terms.py       术语表：字面 + 拼音       polish.py  LLM 同音校订（拼音硬约束）
```

MIT License.
