# 进度账本 —— 2026-08-05-accuracy-and-oss

基线 commit: 7c2517e

Task 1: complete (commits 601a376..HEAD, AST 逐函数比对 31/31 一致；**真机冒烟未做，欠 GPU**)

## GPU 待办（用户 GPU 忙，全部挂起）
- [ ] Task 1 Step 7：真机冒烟，确认拆包后命令行行为不变
- [ ] Task 8 Step 1/6：FunASR 输出格式探测 + 引擎真机验证
- [ ] Task 12：吞吐基准，下限 7.98x
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
