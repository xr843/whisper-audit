# 进度账本 —— 2026-08-05-accuracy-and-oss

基线 commit: 7c2517e

Task 1: complete (commits 601a376..HEAD, AST 逐函数比对 31/31 一致；**真机冒烟未做，欠 GPU**)

## GPU 待办（用户 GPU 忙，全部挂起）
- [ ] Task 1 Step 7：真机冒烟，确认拆包后命令行行为不变
- [ ] Task 8 Step 1/6：FunASR 输出格式探测 + 引擎真机验证
- [x] Task 12：吞吐基准已跑，24.5x（旧 8.4x 是 GPU 忙时测的，已修正），下限 23.3x
- [ ] Task 7/9/11 的 CER 增量实测（需跑流水线产出稿件）
Task 2: complete (commit aca9bd7, 31 tests, 审查未单独跑——并入最终整支审查)
Task 3: complete (commit e5bc004, 39 tests, 审查进行中)
Task 13: 部分完成 (LICENSE/CI/术语表脱敏/README 骨架已提交；**评测小节欠 Task 4**)
  - Task 1 审查结论：规格 ✅ 质量 Approved，无 Critical/Important
  - 已知局限待记录：evaluate.normalize 数字逐字符映射，"100"→"一〇〇" 而非 "一百"
Task 6: complete (commit 84dfcef, 47 tests, 含自加的过度触发验证；待审查)
Task 3 审查结论：规格 ✅，质量「需修改」——1 Critical + 3 Important，已独立复现，fix agent 进行中
  Critical: pinyin_key 对非中文字符返回 ()，两个空元组相等 → APP→ABC 被判 100% 同音，
            静默抬高 homo_pct（决定路线取舍的指标）
虚惊澄清：旧 terms/finance-lecture.json 曾被 Task 3 审查者的 `git checkout aca9bd7 -- .` 复活，
          已确认未进入任何提交，工作区与 HEAD 均干净
Task 1 Step 7: 完成（GPU 空闲后补跑）——薄入口端到端通过，输出数字与拆包前逐项一致
  22 段/610 字、覆盖 67.8%、有效语音 56.4%、34 条字幕、6 段/640 字
Task 2 真机验证: audio-transcribe run 路径与 python3 transcribe.py 产出一致
拼音纠错实测: 见 docs/measurements.md（证实价值 + 发现跨词边界危险，已锁死测试）
Task 4:  complete (commit 9b6c4e1, goldset/eval 全链路实跑验证, 63 tests)
Task 4b: complete (commit a5e25c2, manifest + AISHELL 获取脚本)
Task 5:  complete (commit a5e25c2, CI 夹具 6 条, 78 tests)
Task 12: complete (commit e2b04f0) — 24.5x，两套脚本互印证
Task 8:  FunASR 模型下载中（944MB，本机走代理约 1-2MB/s）

Task 7:  complete (commit 9ead3ee，722f549 是 amend 前的悬空对象) — 拼音纠错接入，记账修了三处
Task 10: complete (commit 27650d3) — polish 拼音硬约束，agent 发现 brief 的示例字对并非同音
Task 4/4b/5/6 审查：2 Critical + 4 Important，均已独立复现
  C1 pinyin_fix 危险是系统性的（节余/结余、空值/控制），**默认已改为关闭**
  C2 evaluate_srt 跨版本对比会因重新切分误报删除错，**已改为按时间重叠选取 + 三列金标**
  I3 pinyin_fix 自身 loose 默认改 False
  I4/I5 账本与计划文档的旧速度数字已同步
  I6 eval --manifest CLI 待接

## 2026-08-05 收尾
Task 9:  complete (commit 95a09d6 + b9ac6e9) — 跨引擎测试 11 条；
         点破 combine 是「逐桶二选一」不是并集，实测 naive 并集更差，保持现状但改口
Task 11: complete (commit 59af6c6) — polish 接入；accurate 档故意未加
文档一致性核查：11 处不一致全部处理 (commit b7ab7b8)
  最险：随包分发的 terms JSON 还在写「零误伤」而代码已默认关闭该功能
最终整支审查：进行中（7c2517e..b7ab7b8，25 commits）

## 尚未完成，且不是代码问题
金标 —— 三条提升路线的开关判定全部悬空。需要一份音频仍在的中文录音，
人工校对 10~15 分钟。归档那份 3.65 小时讲座的源音频已丢失，无法回溯。
