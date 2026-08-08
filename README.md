# WhisperAudit

**[English](https://github.com/xr843/whisper-audit/blob/master/README.en.md) · [中文](https://github.com/xr843/whisper-audit/blob/master/README.md)**

[![PyPI](https://img.shields.io/pypi/v/whisper-audit)](https://pypi.org/project/whisper-audit/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xr843/whisper-audit/blob/master/examples/colab_demo.ipynb)
[![CI](https://github.com/xr843/whisper-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/xr843/whisper-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/xr843/whisper-audit/blob/master/LICENSE)

中文长音频转录流水线，输出文稿、字幕与质检报告。设计目标是**完整性**：
不只转录一遍，还会审计转录结果本身——按实际音量定位「有语音、无文字」的
区段，仅对问题区段定点补转，并内置字级 CER 评测，使准确率可以在使用者
自己的录音上被量化验证。

**在线试用**：点击上方 Open in Colab 徽章，可在浏览器中转录一段真实录音
（免费 T4 GPU，实测 30.8x 实时），无需本地环境。

## 功能特性

- **覆盖率审计**——基于音量分析定位漏转区段，包括隐藏在长片段内部的漏转
- **定点补转**——仅重转问题区段，不做全量重跑
- **三引擎可选**——faster-whisper、FunASR Paraformer、Qwen3-ASR，六个公开域的实测数据见下
- **说话人分离**（可选）——声纹聚类标注 `S1`/`S2`，仅添加标签，绝不改动正文
- **术语表纠错**——字面替换与拼音模糊匹配，所有自动改动带时间戳记账，可逐条回听核对
- **内置评测工具**——`goldset` / `eval` 子命令在自有录音上产出 CER 及错误分解
- **质检报告**——覆盖率、剔除的幻觉、残余可疑段、改动清单，随每次转录输出
- **完全离线**——音频与文本默认不产生任何网络传输（唯一例外 `--polish`，默认关闭）
- **三种使用方式**——命令行、本地网页界面、Colab

## 实测数据（2026-08-07）

六个域 × 三个引擎，同一评测口径（字级 CER，越小越好）。
前三列为干净语音，后三列为困难语音：

| 引擎 | 演讲 | 讲课 | 朗读 | 台湾腔+快语速 | 相声 | 直播带货 |
|---|---|---|---|---|---|---|
| `--engine funasr` | **2.06** | **2.44** | 5.36 | 24.88 | 32.76 | 25.69 |
| `--engine qwen` | 2.11 | 2.61 | **3.92** | 33.99 | 15.05 | 27.10 |
| 默认 whisper | 4.18 | 8.67 | 4.45 | **24.63** | **11.86** | **14.78** |

表中模型：whisper 为 faster-whisper large-v3，funasr 为
SeacoParaformer large，qwen 为 Qwen3-ASR-0.6B。
语料为 SpeechIO 五个测试集与 FLEURS（商用 API 在 SpeechIO 干净集的公开成绩约
1.5~3%）。结论：干净普通话上 `funasr` 最准且删除错少 6~10 倍；困难语音上
whisper 在全部三个域领先，另两个引擎各存在断崖式退化。**默认引擎因此选择
六域中唯一从不崩溃的 whisper，而非平均分最高者。**

消融实验（同音频同参考）显示：干净普通话上审计与补转的收益为零，双路合并
（曾经的默认档）反而使 CER 升高 2.3 倍，默认档已因此改为单路；困难音频上
双路将 CER 从 88.6% 降至 74.2%，`--profile meeting` 保留用于已确认存在漏转
的场景。全部数据（包括对本项目不利的结果与被推翻的结论）见
[docs/measurements.md](https://github.com/xr843/whisper-audit/blob/master/docs/measurements.md)。

| 速度 | 实测值 |
|---|---|
| 长音频（GPU） | 默认档 24.5x、`--profile fast` 62x 实时（RTX 4060 Laptop 8GB）；Colab T4 30.8x |
| 纯 CPU | 3.2x 实时（`--engine funasr --device cpu`），无需独立显卡 |

## 系统要求

| 项目 | 要求 |
|---|---|
| Python | ≥ 3.10 |
| ffmpeg | 必需（音频转码），需通过系统包管理器安装 |
| 操作系统 | Linux 与 WSL2 已实测；macOS 预期可用（CPU 路径）；Windows 原生未测试 |
| GPU | 可选。NVIDIA GPU（8GB 显存已实测）；无 GPU 时可用 CPU 路径 |
| 磁盘 | 每个引擎的模型约 1~3GB，首次运行自动下载 |

## 安装

```bash
pip install "whisper-audit[whisper,cuda]"   # GPU；纯 CPU 环境省略 cuda
sudo apt install ffmpeg                     # macOS: brew install ffmpeg
```

可选依赖按引擎与功能划分，可组合安装：

| Extra | 内容 |
|---|---|
| `whisper` | faster-whisper 引擎（默认引擎） |
| `cuda` | CUDA 12 运行时库。仅 whisper 引擎的 GPU 用户需要；已安装 torch 的环境无需此项 |
| `funasr` | FunASR Paraformer 引擎 |
| `qwen` | Qwen3-ASR 引擎。注意：其依赖将 `transformers` 锁定为 4.57.6（会降级已安装版本），建议独立虚拟环境 |
| `ui` | 本地网页界面 |
| `socks` | SOCKS 代理环境下载模型所需 |

安装注意事项（均来自干净环境实测）：

- 仅安装 `[whisper]` 而缺少 CUDA 运行时的 GPU 环境会在转录开始时报
  `Library libcublas.so.12 is not found`，补装 `[cuda]` 即可
- SOCKS 代理环境下载模型缺少 `socksio` 时会报
  `Using SOCKS proxy, but the 'socksio' package is not installed`，补装 `[socks]` 即可

支持 mp3 / m4a / wav 等常见格式（由 ffmpeg 统一转为 16kHz 单声道）。

## 使用

```bash
# 标准普通话（演讲 / 讲课 / 会议）：该域实测最准且最快
#   引擎需对应 extra：pip install "whisper-audit[funasr]"
whisper-audit run 录音.mp3 --engine funasr

# 通用场景（默认配置）：单路转录 + 覆盖率审计 + 定点补转
whisper-audit run 录音.mp3

# 已确认存在漏转的困难音频（强噪声 / 拖腔）：双路交叉
whisper-audit run 录音.mp3 --profile meeting

# 速度优先：turbo 模型 62x 实时，实测质量代价约 0.9 个百分点
whisper-audit run 录音.mp3 --profile fast

# 多说话人：标注说话人（仅添加标签，不改动正文）
whisper-audit run 访谈.mp3 --engine funasr --diarize

# 本地网页界面：浏览器中上传音频、查看结果、下载产物
#   安装：pip install "whisper-audit[whisper,ui]"
whisper-audit ui
```

网页界面默认仅绑定 `127.0.0.1`。以 `whisper-audit ui --listen 0.0.0.0` 启动时，
局域网内其他设备可通过浏览器直接使用本机的转录服务（客户端无需安装任何组件）；
此模式下音频将经局域网传输至运行服务的机器，请按环境的安全要求决定是否启用。

![whisper-audit ui：左侧上传与进度，右侧质检摘要与正文预览](https://raw.githubusercontent.com/xr843/whisper-audit/master/docs/images/webui.jpg)

（截图为真实运行：Colab demo 同款《阿Q正传》朗读，本次 26.1x 实时）

源码方式运行：`python3 transcribe.py 录音.mp3`，与安装后的命令等价。
其余参数（`--terms`、`--keep-break`、`--speakers`、`--model`、`--device`、
`--language`、`--title`、`-o`）见 `--help`。

### 引擎选择

- 清晰普通话 → `funasr`
- 录音质量好的混合体裁**短**音频 → `qwen`（长音频实测仅 0.4~1.2x 实时，不适用）
- 口音重、噪声大、业余或远场录音、无法预判类型 → 默认 whisper。
  实测低音量带混响的业余朗读可使 funasr 退化至 35% CER，whisper 同音频为 13%

### 输出

默认写入音频同目录的 `<名称>_转录/`：

| 文件 | 内容 |
|---|---|
| `*_全文转录.md` / `.txt` | 正文，时间戳分段、自动标点；启用 `--diarize` 时换人处分段并标注 `S1`/`S2` |
| `*_字幕.srt` | 按词级时间戳切分的字幕，可对照原音频逐句核对；启用 `--diarize` 时每条带说话人标签 |
| `质检报告.json` | 覆盖率、漏转区段、幻觉清单、自动改动记账，建议交付前检查 |
| `.work/` | 中间结果，重跑时自动复用（更改参数前应删除） |

日志末尾的「终审」行报告最终成稿的实际覆盖率与残余可疑段数量。
以 LibriVox 公有领域朗读《阿Q正传》第一章（7.8 分钟，与 Colab demo 同款
音频）的一次真实运行为例：

```
审计：覆盖 95.7%　讲话中位音量 -27.2dB　待补 0 处（其中段内饥饿 0）　幻觉 0 处
终审：合并稿覆盖 95.7%　有效语音 95.7%　残余饥饿段 0 处
完成：6 段 / 1,646 字 / 64 条字幕
```

唯一未覆盖区段是末尾 5.1 秒——质检报告给出其精确区间，音量测量判定其中
无语音，故不补转。

## 评测自己录音的准确率

```bash
# 1. 从转录输出中切出一段，生成待校对稿（初始内容即 ASR 结果）
whisper-audit goldset 输出目录/ --from 00:10:00 --to 00:20:00 -o sample.gold.tsv
# 2. 编辑 sample.gold.tsv：仅修改第三列中的错字，不改动时间戳、不合并行
# 3. 评测
whisper-audit eval --gold sample.gold.tsv --hyp 输出目录/
```

报告包含 CER、**删除率**（衡量漏转程度，本项目的核心指标）与**同音/近音错误
占比**（决定拼音类纠错手段的理论上限）。`eval --manifest` 亦可直接评测公开
基准集。

## 术语表（可选）

```json
{ "name": "领域名",
  "fixes": [["误识写法", "正确写法"]],
  "terms": ["正确写法"] }
```

- `fixes`：字面精确替换，建议逐条人工确认；工具会统计每条的命中次数，
  便于发现从未命中或命中异常偏多的条目
- `terms`：拼音模糊匹配，一条覆盖一类同音错误。默认关闭，`--pinyin-fix` 显式启用

示例：`examples/terms/finance-lecture.json`（42 条 fixes + 41 条 terms，
经真实录音核定）。

## 会修改正文的可选功能（默认全部关闭）

| 参数 | 作用 | 默认关闭的原因 |
|---|---|---|
| `--pinyin-fix` | 拼音级术语纠错 | 无调拼音会将「节余/结余」等真实存在、含义不同的词判为同音 |
| `--polish` | LLM 同音校订 | 正文将发送至所配置的 endpoint；且同音替换本身可能改变语义 |

启用后所有改动均记入质检报告（带时间戳，可回听核对），应逐条检查。
`--polish` 配套 `--llm-base-url` / `--llm-model` / `--polish-dry-run`，
API key 仅从环境变量 `WHISPER_AUDIT_LLM_KEY` 读取。其护栏强制每处替换
与原文真同音、行数不变，内容无法被增删——但同音替换本身仍可能改变
语义，故保持默认关闭。

## 已知局限

1. ASR 准确率存在上限：方言口音与术语密集段落错误集中，术语表可修复高频错误，
   其余需对照 `.srt` 人工校对
2. 标点为推定结果（基于词间停顿与连接词），不代表讲者原意
3. 说话人标签为声纹聚类推定：短促插话可能归属错误；仅识别出一人时不添加标签；
   该功能从不改动正文（有测试锁定此不变量）
4. 歌唱类音频不适用（慢速拖腔实测 CER 74% 以上，删除错误主导，补转无法恢复）
5. 中场休息自动识别可能误判，可用 `--keep-break` 关闭

## 文档

- [docs/lessons.md](https://github.com/xr843/whisper-audit/blob/master/docs/lessons.md) —— 24 条实测工程记录：VAD 误删、静音幻觉、
  段内漏转、合并陷阱、被污染的性能数据等，每条均来自真实测量
- [docs/measurements.md](https://github.com/xr843/whisper-audit/blob/master/docs/measurements.md) —— 全部实测数字与每个默认值的
  判定依据，被推翻的结论原样保留
- [Releases](https://github.com/xr843/whisper-audit/releases) —— 版本变更记录

## 参与开发

```bash
pip install -e ".[dev]"       # 纯 CPU 核心依赖，不含 ASR 后端
python3 -m pytest tests/ -q   # 秒级完成，无需 GPU 与音频文件
```

测试套件刻意不依赖 GPU 与音频：引擎适配器是对真机探测输出的纯函数，
说话人分离测试使用保存为夹具的真实声纹向量。测试保护的是那类不抛异常的
缺陷——静默丢失内容、静默改写文本、静默虚报覆盖率。

开发环境、测试纪律与当前最需要的贡献方向见 [CONTRIBUTING.md](https://github.com/xr843/whisper-audit/blob/master/CONTRIBUTING.md)。

```
whisper_audit/
  cli.py         命令行与主流程          engines/   ASR 后端（whisper/funasr/qwen）
  audio.py       转码 + 音量测量          audit.py   空洞 / 幻觉 / 段内饥饿 / 终审
  merge.py       多路合并与去重           render.py  标点 / 分段 / 出稿 / 字幕重切
  evaluate.py    字级 CER 与错误分解      goldset.py 待校对稿生成与评测入口
  terms.py       术语表：字面 + 拼音       polish.py  LLM 同音校订（拼音硬约束）
  diarize.py     声纹抽取与说话人聚类     webui.py   本地网页界面
```

## 致谢

本项目基于以下开源工作构建：

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / CTranslate2 与 OpenAI Whisper 模型
- [FunASR](https://github.com/modelscope/FunASR)（SeacoParaformer 与 cam++ 声纹模型）
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Gradio](https://github.com/gradio-app/gradio)（网页界面）
- 评测语料：[SpeechIO Leaderboard](https://github.com/SpeechColab/Leaderboard)、
  [FLEURS](https://huggingface.co/datasets/google/fleurs)；
  演示音频来自 [LibriVox](https://librivox.org/)（公有领域）

## 许可证

[MIT](https://github.com/xr843/whisper-audit/blob/master/LICENSE)
