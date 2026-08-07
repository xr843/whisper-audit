# 参与开发 / Contributing

English below.

## 这个项目的一条铁律

**任何会改动转录正文的行为，都必须能被关掉、被记账、被复现。**

流水线里绝大多数 bug 不会抛异常——它们静悄悄吃掉内容、静悄悄改写文字、
静悄悄虚报覆盖率，成品看着完全正常。所以：

- 新增任何自动修改正文的功能，**默认必须是关的**，直到有 CER 数据证明净收益为正
- 每一处改动都要进 `质检报告.json` 的记账，带时间戳可回听
- 「测不出收益的默认关掉」不是口号，是 `docs/measurements.md` 里逐条执行的纪律

## 开发环境

```bash
pip install -e ".[dev]"        # 纯 CPU 核心依赖，不装任何 ASR 后端
python3 -m pytest tests/ -q    # 秒级，无需 GPU 与音频文件
```

测试刻意不依赖 GPU 和音频：引擎适配器是对**真机探测下来的输出**做的纯函数，
说话人分离测试用的是存成夹具的真实 cam++ 声纹。这样 CI 装得动、跑得快，
而它保护的恰恰是那一类永远不报错的 bug。

## 提 PR 之前

1. **`python3 -m pytest tests/ -q` 全绿。**
2. **改了行为就要有测试，而且要验证它真能抓到。** 方法很简单——把你的修复
   撤掉，跑测试，它必须失败。测试和实现共享同一套错误假设是常事，
   不做这一步就不知道测的是什么。
3. **改了默认值、阈值、参数，必须在 `docs/measurements.md` 里附上判定依据。**
   包括标定数据来自哪个域——`docs/lessons.md` 第 17 条记的就是一个在失效域
   标出来的阈值，看起来和任何别的实测数字一样可信，实际让整个功能失效。
4. **性能数字要在 GPU 空闲时测。** `nvidia-smi` 确认一下，第 9 条坑记过
   一次被污染的读数。

## 什么样的贡献最有价值

按当前的缺口排序：

1. **困难域实测**：重口音、强噪声、方言。`qwen` 引擎在这些域一个数都没有，
   这也是它没被设为默认的唯一原因。
2. **多语言**：`whisper` 和 `qwen` 本身就多语言，卡住的是流水线里的中文硬编码
   （`SPEECH_RATE` 字/秒、繁简转换、拼音模块）。
3. **合并策略**：`combine()` 是逐 30 秒桶取字数多的一路整窗胜出。
   「字多者胜」在两个引擎错误模式不同时会失效（见 measurements.md 的跨引擎裁决）。
4. **任何一条踩坑记录**：`docs/lessons.md` 是这个项目最有价值的部分。
   你踩到的坑，比新功能值钱。

## Bug 报告

请附上：完整命令行、`质检报告.json`、日志里的「审计」与「终审」两行。
音频不用给——那两行数字通常就够定位。

---

# Contributing (English)

## The one hard rule

**Anything that modifies transcript text must be switchable, accounted for, and
reproducible.**

Most bugs in this pipeline never raise an exception. They silently drop content,
silently rewrite words, or silently overstate coverage — and the output looks
perfectly fine. Therefore:

- Any new feature that edits text **ships disabled by default** until CER data
  shows it is a net win
- Every edit is logged to `质检报告.json` with a timestamp you can listen back to
- "If it doesn't measure a gain, it stays off" is enforced case by case in
  `docs/measurements.md`

## Setup

```bash
pip install -e ".[dev]"        # pure-CPU core deps, no ASR backend
python3 -m pytest tests/ -q    # seconds, no GPU or audio required
```

Tests deliberately avoid GPU and audio: engine adapters are pure functions over
recorded real-world outputs, and diarization tests run against real cam++
embeddings saved as a fixture.

## Before opening a PR

1. **`python3 -m pytest tests/ -q` passes.**
2. **Behaviour changes need a test — and verify the test actually catches the
   bug.** Revert your fix, run the test, confirm it fails. Tests and
   implementations routinely share the same wrong assumption; without this step
   you don't know what you're testing.
3. **Changed a default, threshold, or parameter? Justify it in
   `docs/measurements.md`**, including which domain the calibration data came
   from. Lesson 17 records a threshold calibrated on a known-failure domain — it
   looked exactly as credible as any other measured number, and it silently
   disabled the whole feature.
4. **Measure performance on an idle GPU.** Check `nvidia-smi` first; lesson 9
   records one contaminated reading.

## Most valuable contributions right now

1. **Hard-domain measurements** — heavy accents, noise, dialects. The `qwen`
   engine has zero numbers there, which is the only reason it isn't the default.
2. **Multilingual support** — `whisper` and `qwen` are already multilingual; what
   blocks it is Chinese-specific logic in the pipeline (`SPEECH_RATE` in
   chars/sec, traditional→simplified conversion, the pinyin module).
3. **Merge strategy** — `combine()` picks the pass with more characters per
   30-second bucket. "More characters wins" breaks down when two engines have
   different failure modes (see the cross-engine verdict in measurements.md).
4. **Any field-tested trap** for `docs/lessons.md`. That file is the most
   valuable thing in this repo. A trap you hit is worth more than a feature.

## Bug reports

Include the full command line, `质检报告.json`, and the two log lines starting
with 审计 (audit) and 终审 (final audit). No audio needed — those numbers are
usually enough.
