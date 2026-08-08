# WhisperAudit

**[English](README.en.md) · [中文](README.md)**

[![PyPI](https://img.shields.io/pypi/v/whisper-audit)](https://pypi.org/project/whisper-audit/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xr843/whisper-audit/blob/master/examples/colab_demo.ipynb)
[![CI](https://github.com/xr843/whisper-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/xr843/whisper-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**免安装试用**：点上面的 Open in Colab 徽章，浏览器里 5 分钟转完一段真实录音——免费 T4 实测 30.8x 实时，无需任何本地环境。

中文长音频 → 文稿/字幕。目标是**不遗漏**，不是「转一遍」：
自动审计漏转、定点补转、可选说话人标注，并给你一把可复现的正确率尺子。
三引擎可选，本机 GPU 或**纯 CPU** 离线运行，音频不出本机。

## 实测成绩（2026-08-07）

六个域 × 三个引擎，同一把尺子（字级 CER，越小越好）。
前三列干净域，后三列困难域：

| 引擎 | 演讲 | 讲课 | 朗读 | 台湾腔+极快 | 相声 | 直播带货 |
|---|---|---|---|---|---|---|
| `--engine funasr` | **2.06** | **2.44** | 5.36 | 24.88 | 32.76 | 25.69 |
| `--engine qwen` | 2.11 | 2.61 | **3.92** | 33.99 | 15.05 | 27.10 |
| 默认 whisper | 4.18 | 8.67 | 4.45 | **24.63** | **11.86** | **14.78** |

语料：SpeechIO 五个集 + FLEURS（商用 API 在 SpeechIO 干净集约 1.5~3%）。
规律一句话：**干净普通话 `funasr` 最准且漏字少 6~10 倍；困难域 whisper 全胜、
另两个各有断崖——默认值留给最稳的，不留给平均分最高的。**

**流水线自己值多少？** 消融实测（同音频同参考）：干净普通话上审计+补转收益
**恰好为零**，双路合并（旧默认）反而把 CER 翻 2.3 倍——默认档已因此改为单路；
困难音频上双路把 88.6% 拉回 74.2%，`--profile meeting` 留作**已知在丢内容时**
的补救手段。不利的数字也全在 [docs/measurements.md](docs/measurements.md)，
含被推翻的结论（原样保留）。

| 速度 | 实测 |
|---|---|
| 长音频（GPU） | 默认档 24.5x / `--profile fast` 62x（RTX 4060 8GB）；Colab 免费 T4 实测 30.8x |
| 纯 CPU | 3.2x（`--engine funasr --device cpu`），普通笔记本即可 |

## 安装

```bash
git clone https://github.com/xr843/whisper-audit && cd whisper-audit
pip install -e ".[whisper,cuda]"     # GPU；纯 CPU 去掉 cuda
sudo apt install ffmpeg              # 系统依赖，pip 装不了；macOS: brew install ffmpeg
```

引擎按需选，可组合：`whisper` / `funasr` / `qwen`。

**两个装完才会发现的坑**（2026-08-07 干净机实测踩到，已写进 extras）：

- **`cuda` 只有 whisper 引擎的 GPU 用户需要**。CTranslate2 不打包 CUDA 运行时，
  只装 `[whisper]` 会在真正开始转录时报 `Library libcublas.so.12 is not found`。
  单独成 extra 是因为这两个 wheel 近 700MB；`funasr` / `qwen` 走 torch，
  torch 自带这些库，**不需要 `cuda`**；已装 torch 的环境同理。
- **`qwen` 是重量级安装**：`qwen-asr` 把 `transformers` 钉成 `==4.57.6`
  （精确等号，会强制降级），并硬依赖 gradio + flask。
  环境里有别的东西依赖新版 transformers 的话，给它单独建一个 venv。
- **走 SOCKS 代理的话加 `socks`**。`huggingface_hub` 改用 httpx 后，
  SOCKS 代理下载模型需要额外的 `socksio`，否则报
  `Using SOCKS proxy, but the 'socksio' package is not installed`。

模型首次运行自动下载（large-v3 约 2.9GB）。mp3 / m4a / wav 等常见格式都支持
（ffmpeg 统一转 16kHz 单声道）。

## 用法

```bash
# 标准普通话（演讲/讲课/会议）—— 质量最高且快：CER 2% 级、漏字少 6~10 倍
whisper-audit run 录音.mp3 --engine funasr

# 干净的混合体裁「短」音频 —— 念稿域最准；长音频勿用（仅 0.4~1.2x 实时）
whisper-audit run 短录音.mp3 --engine qwen

# 拿不准 —— 默认档：单路 + 覆盖率审计 + 定点补转
whisper-audit run 录音.mp3

# 已经发现单路在丢内容（强噪声 / 拖腔 / 极端口音）—— 双路交叉补救
whisper-audit run 录音.mp3 --profile meeting

# 赶时间 —— turbo 引擎 62x 实时，质量代价实测仅 0.9pp
whisper-audit run 录音.mp3 --profile fast

# 多人对话 —— 标注说话人（只加标签，不改一个字；三个引擎都能配）
whisper-audit run 访谈.mp3 --engine funasr --diarize

# 图形界面 —— 给不用终端的人：浏览器里拖音频进来
whisper-audit ui          # 需 pip install "whisper-audit[ui]"；只绑本机，音频不出电脑
```

`whisper-audit ui --listen 0.0.0.0` 可让局域网同事直接用浏览器访问你这台机器转录
（他们什么都不用装）——但音频会经过内网传到你机器，想清楚再开。

`python3 transcribe.py 录音.mp3` 与装包后的命令等价。
无显卡：`--engine funasr --device cpu`（3.2x 实时）。
其他参数：`--terms 术语表.json`、`--keep-break`（不剔除中场休息）、
`--speakers N`（已知人数）、`--model`、`--device`、`--language`、`--title`、`-o`，
详见 `--help`。

**引擎怎么选**：清晰普通话 → `funasr`；干净混合体裁**短**音频 → `qwen`；
口音重 / 噪声大 / **业余、远场录音** / 拿不准 → 默认 whisper（六个域里唯一从不崩的，见上表。实测连音量低、带混响的业余朗读都能让 funasr 崩到 35% CER，whisper 同音频 13%）。

### 输出

默认写到音频同目录的 `<名称>_转录/`：

| 文件 | 用途 |
|---|---|
| `*_全文转录.md` / `.txt` | 正文，时间戳分段、自动标点；`--diarize` 时换人处断段并标 `S1`/`S2` |
| `*_字幕.srt` | 词级时间戳切分，配原音频逐句回听核对；`--diarize` 时每条带 `[S1]` |
| `质检报告.json` | 覆盖率、漏转区段、幻觉清单、改动记账——**出稿前看一眼** |
| `.work/` | 中间结果，重跑自动复用（改参数前先删） |

日志末尾的「终审」行报告合并稿的实际覆盖率与残余可疑段。

## 测一测你这份录音转得多准

```bash
# 1. 从输出切一段生成待校对稿（初稿就是 ASR 结果）
whisper-audit goldset 输出目录/ --from 00:10:00 --to 00:20:00 -o sample.gold.tsv
# 2. 打开 sample.gold.tsv，只改第三列的错字——不动时间戳、不合并行
# 3. 评测
whisper-audit eval --gold sample.gold.tsv --hyp 输出目录/
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
key 只从环境变量 `WHISPER_AUDIT_LLM_KEY` 读。

## 已知局限

1. **ASR 准确率有上限**：方言口音 + 术语密集段落错得密，术语表修高频错，
   其余需对照 `.srt` 回听校对
2. **标点是推定的**（词间停顿 + 连接词），不代表讲者原意
3. **说话人分离是可选标注**（`--diarize`，默认关）：标签靠声纹聚类推定，
   短促插话（「嗯」「对」）容易归错人；只认出一人时不标。它**从不改动正文**
4. **歌唱类音频不适用**（慢速拖腔实测 CER 74%+，删除主导，补转救不回）
5. 中场休息自动识别可能误判，`--keep-break` 可关

## 深入阅读

- **[docs/lessons.md](docs/lessons.md)** —— 24 条实测踩坑记录：VAD 误杀、静音幻觉、
  段内饥饿、合并陷阱、性能真相……每一条都是测出来的，不是文档推理。
  **这份记录是本项目真正的价值所在。**
- **[docs/measurements.md](docs/measurements.md)** —— 全部实测数字与每个默认值的
  判定依据，包括被推翻过的结论（原样保留）

## 参与开发

欢迎 PR。开发环境、测试纪律与当前最需要的贡献方向见
[CONTRIBUTING.md](CONTRIBUTING.md)。


```bash
pip install -e ".[dev]"       # 纯 CPU 核心依赖，不装 ASR 后端
python3 -m pytest tests/ -q   # 218 tests + 1 xfail，秒级，无需 GPU/音频
```

```
whisper_audit/
  cli.py         命令行与主流程          engines/   ASR 后端（whisper/funasr/qwen）
  audio.py       转码 + 音量测量          audit.py   空洞 / 幻觉 / 段内饥饿 / 终审
  merge.py       多路合并与去重           render.py  标点 / 分段 / 出稿 / 字幕重切
  evaluate.py    字级 CER 与增删改分解     goldset.py 待校对稿生成与评测入口
  terms.py       术语表：字面 + 拼音       polish.py  LLM 同音校订（拼音硬约束）
  diarize.py     声纹抽取与说话人聚类
```

MIT License.
