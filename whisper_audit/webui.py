"""本地网页界面：给不用终端的人。

设计约束（每条都有血泪出处）：

1. **转录必须跑在子进程里**，绝不能在 UI 进程内直接调 cmd_run——
   `ensure_cuda_libs()` 设好 LD_LIBRARY_PATH 后会 `os.execv` 重启进程，
   在 UI 进程里调用等于把网页服务当场杀掉（lessons 第 21 条的同族坑）。
   子进程用 `[sys.executable, "-m", "whisper_audit.cli", ...]`——
   这条 `-m` 重启路径已被修复并有测试锁着。

2. **只绑 127.0.0.1**。「音频不出本机」是本项目对用户的承诺，
   网页壳不能悄悄把它变成一个公网服务。

3. gradio 是重依赖，只在 `launch()` 里 import——测试与 CI 不装它
   也能测本模块的纯函数部分。
"""
import json
import os
import subprocess
import sys

# 展示名 → CLI 参数。顺序即下拉框顺序，默认第一项。
ENGINES = [
    ("whisper —— 拿不准就选它（噪声/口音/业余录音都不崩）", "whisper"),
    ("funasr —— 清晰普通话最准最快（录音质量差会崩，慎选）", "funasr"),
    ("qwen —— 仅限短音频（长音频慢于实时）", "qwen"),
]
PROFILES = [
    ("标准（单路+审计+补转）", "lecture"),
    ("补救（双路合并——只在确认丢内容时用，干净音频会更差）", "meeting"),
    ("求快（turbo，质量代价约 1 个百分点）", "fast"),
]


def build_cmd(audio, outdir, engine="whisper", profile="lecture",
              terms=None, diarize=False, speakers=None):
    """拼子进程命令。纯函数，可测。"""
    cmd = [sys.executable, "-m", "whisper_audit.cli", "run", audio,
           "-o", outdir, "--engine", engine, "--profile", profile]
    if terms:
        cmd += ["--terms", terms]
    if diarize:
        cmd += ["--diarize"]
        if speakers:
            cmd += ["--speakers", str(int(speakers))]
    return cmd


def collect_outputs(outdir):
    """转录产物清单：(正文md, 正文txt, 字幕srt, 质检json)，缺哪个哪个为 None。"""
    found = {"md": None, "txt": None, "srt": None, "qa": None}
    if not os.path.isdir(outdir):
        return found
    for name in sorted(os.listdir(outdir)):
        p = os.path.join(outdir, name)
        if name.endswith("_全文转录.md"):
            found["md"] = p
        elif name.endswith("_全文转录.txt"):
            found["txt"] = p
        elif name.endswith("_字幕.srt"):
            found["srt"] = p
        elif name == "质检报告.json":
            found["qa"] = p
    return found


def qa_summary(qa_path):
    """质检报告 → 给普通人看的几行结论。读不出来就如实说，不编。"""
    try:
        r = json.load(open(qa_path, encoding="utf-8"))
        f = r["final"]
    except Exception as e:
        return f"（质检报告读取失败：{type(e).__name__}）"
    lines = [f"时间覆盖率 {f['cover_pct']:.1f}%　有效语音 {f['speech_pct']:.1f}%",
             f"残余可疑段 {f['starved']} 处　剔除幻觉 {len(r.get('hallucinations', []))} 处"]
    fixes = r.get("pinyin_fixes") or []
    hits = r.get("terms_hits") or {}
    if hits:
        lines.append("术语修正：" + "、".join(f"{k}×{v}" for k, v in hits.items() if v))
    if fixes:
        lines.append(f"拼音级纠错 {len(fixes)} 处（详单在质检报告.json）")
    if f["starved"]:
        lines.append("⚠ 有段落时长撑不起字数——出稿前对照字幕回听这些位置")
    return "\n".join(lines)


def transcribe_stream(audio, engine_label, profile_label, terms, diarize, speakers):
    """gradio 生成器：流式吐日志，最后给出产物。"""
    if not audio:
        yield "请先选择音频文件", None, None, None, ""
        return
    engine = dict(ENGINES).get(engine_label, "whisper")
    profile = dict(PROFILES).get(profile_label, "lecture")
    outdir = os.path.splitext(audio)[0] + "_转录"
    cmd = build_cmd(audio, outdir, engine, profile,
                    terms=terms or None, diarize=diarize, speakers=speakers)

    logs = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        # 进度条控制字符和框架噪声不进 UI
        if not line or "\r" in line or "it/s" in line:
            continue
        logs.append(line)
        yield "\n".join(logs[-30:]), None, None, None, ""
    proc.wait()

    out = collect_outputs(outdir)
    if proc.returncode != 0 or not out["txt"]:
        logs.append(f"✗ 转录失败（退出码 {proc.returncode}）——把上面的日志发给管理员")
        yield "\n".join(logs[-30:]), None, None, None, ""
        return
    body = open(out["txt"], encoding="utf-8").read()
    files = [p for p in (out["md"], out["txt"], out["srt"], out["qa"]) if p]
    summary = qa_summary(out["qa"]) if out["qa"] else "（无质检报告）"
    logs.append("✅ 完成")
    yield "\n".join(logs[-30:]), body[:8000], files, out["srt"], summary


def exempt_localhost_from_proxy(environ):
    """把 localhost/127.0.0.1 加进 no_proxy。纯函数（就地改传入的映射并返回）。

    代理环境（国内常态）下，gradio 启动时对 http://127.0.0.1 的自检请求会被
    系统代理劫走而失败，它随即报
    `ValueError: When localhost is not accessible, a shareable link must be
    created. Please set share=True` ——而 share=True 是公网分享，
    与「音频不出本机」的承诺相反。正确修法是豁免 localhost，不是开 share。
    2026-08-08 本机实测踩到。
    """
    add = ["localhost", "127.0.0.1", "::1"]
    for key in ("no_proxy", "NO_PROXY"):
        cur = [x for x in environ.get(key, "").split(",") if x]
        environ[key] = ",".join(cur + [a for a in add if a not in cur])
    return environ


def supported_kwargs(fn, **kw):
    """按 fn 的签名过滤 kwargs。纯函数，可测。

    ui extra 只约束 gradio>=4：干净安装拿到 6.x，被 qwen-asr 连带装过的
    环境可能是 5.x——两代 API 各有增删（6.x 删了 Textbox 的
    show_copy_button、launch 的若干参数）。展示性参数宁可静默丢弃，
    不能让整个 UI 起不来。"""
    import inspect
    try:
        names = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return kw
    return {k: v for k, v in kw.items() if k in names}


def launch(port=7860, server_name="127.0.0.1"):
    exempt_localhost_from_proxy(os.environ)
    # 与「音频不出本机」同一立场：本地工具不发遥测
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    with gr.Blocks(title="whisper-audit 本地转录") as app:
        gr.Markdown("# whisper-audit 本地转录\n"
                    "音频只在**这台电脑**上处理，不上传到任何地方。"
                    "转录完成后正文可直接复制，四个文件可下载。")
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.File(label="音频文件（mp3 / m4a / wav 均可）",
                                file_types=["audio"], type="filepath")
                engine = gr.Dropdown([l for l, _ in ENGINES], value=ENGINES[0][0],
                                     label="引擎")
                profile = gr.Dropdown([l for l, _ in PROFILES], value=PROFILES[0][0],
                                      label="档位")
                with gr.Accordion("高级选项", open=False):
                    terms = gr.File(label="术语表 json（可选）", type="filepath")
                    diarize = gr.Checkbox(label="标注说话人（多人对话时勾选）")
                    speakers = gr.Number(label="已知说话人数（留空自动判断）",
                                         value=None)
                btn = gr.Button("开始转录", variant="primary")
                log = gr.Textbox(label="进度", lines=12, interactive=False)
            with gr.Column(scale=2):
                summary = gr.Textbox(label="质检摘要（先看这里）", lines=5,
                                     interactive=False)
                body = gr.Textbox(label="正文（前 8000 字预览，完整版下载 txt/md）",
                                  lines=22, interactive=False)
                files = gr.File(label="下载：正文 md / txt · 字幕 srt · 质检报告",
                                file_count="multiple", interactive=False)
                srt = gr.File(label="字幕单独下载", interactive=False)
        btn.click(transcribe_stream,
                  inputs=[audio, engine, profile, terms, diarize, speakers],
                  outputs=[log, body, files, srt, summary])
    app.launch(share=False, **supported_kwargs(
        app.launch, server_name=server_name, server_port=port,
        inbrowser=True, quiet=True))
