"""命令行入口。"""
import argparse
import json
import os

from . import hms, log
from .audio import Loudness, ensure_cuda_libs, prepare_audio
from .audit import SPEECH_RATE, audit, audit_rows, find_breaks, in_any
from .engines.whisper import repatch, transcribe_pass
from .merge import combine
from .render import raw_text, render, terms_hits

# ---------------------------------------------------------------- 参数档

# batch=16 必须配 int8_float16：8GB 卡上 fp16+batch16 会 OOM（实测）。
# int8 本身不提速，它的价值就是省出显存来开大 batch —— 那才是提速来源。
PROFILES = {
    # 单人讲授、音质好：单路 + 补转即可，最快
    "lecture": {"two_pass": False, "chunk_coarse": 30, "batch": 16, "beam": 5,
                "compute": "int8_float16"},
    # 多人问答、口音重、内容重要：双路交叉，最全
    "meeting": {"two_pass": True, "chunk_coarse": 30, "chunk_fine": 10, "batch": 16, "beam": 5,
                "compute": "int8_float16"},
    # 只求快（质量有代价，专业术语多的内容不建议）
    "fast": {"two_pass": False, "chunk_coarse": 30, "batch": 16, "beam": 1,
             "compute": "int8_float16"},
}


# ---------------------------------------------------------------- 主流程

def main(argv=None):
    ap = argparse.ArgumentParser(description="长音频转录流水线（以不遗漏为目标）")
    ap.add_argument("audio")
    ap.add_argument("-o", "--outdir", default=None, help="输出目录，默认与音频同名")
    ap.add_argument("--profile", default="meeting", choices=list(PROFILES))
    ap.add_argument("--terms", default=None, help="术语修正表 json")
    ap.add_argument("--title", default=None)
    ap.add_argument("--keep-break", action="store_true", help="不剔除中场休息段")
    ap.add_argument("--model", default="large-v3", help="faster-whisper 模型名")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--compute", default=None, help="覆盖档位里的 compute_type")
    ap.add_argument("--language", default="zh")
    args = ap.parse_args(argv)

    ensure_cuda_libs()

    src = os.path.abspath(args.audio)
    title = args.title or os.path.splitext(os.path.basename(src))[0]
    outdir = os.path.abspath(args.outdir or os.path.join(os.path.dirname(src), title + "_转录"))
    work = os.path.join(outdir, ".work")
    os.makedirs(work, exist_ok=True)

    terms = json.load(open(args.terms, encoding="utf-8")) if args.terms else {}
    cfg = PROFILES[args.profile]
    compute = args.compute or (cfg["compute"] if args.device != "cpu" else "int8")
    mk = dict(model_name=args.model, device=args.device, language=args.language)
    log(f"档位 {args.profile}　模型 {args.model}／{args.device}／{compute}　输出 {outdir}")

    wav = prepare_audio(src, work)
    loud = Loudness(wav)

    passes = [transcribe_pass(wav, os.path.join(work, "pass1.json"),
                              cfg["chunk_coarse"], cfg["batch"], cfg["beam"], compute, **mk)]
    if cfg["two_pass"]:
        passes.append(transcribe_pass(wav, os.path.join(work, "pass2.json"),
                                      cfg["chunk_fine"], cfg["batch"], cfg["beam"],
                                      compute, **mk))

    dur = passes[0]["duration"]
    rep = audit(passes[0], loud)
    n_starved = sum(1 for _, _, lab in rep["spans"] if lab == "段内饥饿")
    log(f"审计：覆盖 {rep['cover_pct']:.1f}%　讲话中位音量 {rep['speech_db']:.1f}dB　"
        f"待补 {len(rep['spans'])} 处（其中段内饥饿 {n_starved}）　幻觉 {len(rep['hallu'])} 处")

    breaks = [] if args.keep_break else find_breaks(passes[0], loud)
    for a, b in breaks:
        log(f"识别到休息段 {hms(a)}–{hms(b)}（{(b-a)/60:.1f} 分钟），将剔除")
    if breaks:
        rep["spans"] = [s for s in rep["spans"] if not in_any(s[0], breaks)]

    patch = repatch(wav, rep["spans"], os.path.join(work, "repatch.json"),
                    compute=compute, **mk)
    rows = combine(passes, patch, terms, breaks, dur, rep.get("drop", []))
    per_pass = ", ".join(
        format(sum(len(s["text"]) for s in p["segments"]), ",") for p in passes)
    log(f"合并后 {len(rows)} 段 / {sum(len(r['text']) for r in rows):,} 字"
        f"（各单路：{per_pass}）")

    # 终审：审的必须是交付物本身。过去这里报的是 pass1 的数（97.3%），
    # 而成品字幕的时间并集只有 90.0%，给读者看的数字对不上交付物。
    fin = audit_rows(rows, dur)
    log(f"终审：合并稿覆盖 {fin['cover_pct']:.1f}%　有效语音 {fin['speech_pct']:.1f}%　"
        f"残余饥饿段 {fin['starved']} 处")
    if fin["starved"]:
        log("  ⚠ 仍有段落时长撑不起字数，补转没能捞回来，出稿前请对照 .srt 回听这些位置")

    # 必须在替换**之前**数：正文里源词早被换掉了，事后统计只会全是 0
    hits = terms_hits(raw_text(passes), terms)
    if hits:
        miss = [k for k, v in hits.items() if v == 0]
        log(f"术语表：{len(hits)} 条，命中 {sum(1 for v in hits.values() if v)} 条"
            + (f"，0 命中 {len(miss)} 条（{'、'.join(miss[:5])}{'…' if len(miss)>5 else ''}）"
               if miss else ""))

    meta_lines = [
        f"**转录方式**　faster-whisper {args.model}，{len(passes)} 路交叉 + "
        f"{len(rep['spans'])} 处定点补转，取并集。\n",
        f"**覆盖率**　本文档时间覆盖 {fin['cover_pct']:.1f}%，其中字数撑得起的有效语音约 "
        f"{fin['speech_pct']:.1f}%（按 {SPEECH_RATE:.0f} 字/秒估）；"
        f"单路原始审计 {rep['cover_pct']:.1f}%，讲话段中位音量 {rep['speech_db']:.1f} dBFS。\n",
    ]
    for a, b in breaks:
        meta_lines.append(f"**已剔除**　休息段 {hms(a)}–{hms(b)}"
                          f"（{(b-a)/60:.1f} 分钟，音量显著低于讲授段，转录输出为幻觉）。\n")
    if rep["hallu"]:
        meta_lines.append(f"**已剔除幻觉片段** {len(rep['hallu'])} 处（静音段被转成网络字幕套语）。\n")
    if fin["starved"]:
        meta_lines.append(f"**待人工核对** {fin['starved']} 处段落时长与字数明显不匹配，"
                          "疑似仍有未转出的内容，请对照字幕回听。\n")
    meta_lines.append("**注意**　语音识别对专有名词与专业术语存在同音误识，"
                      "已按术语表统一校正可确定者，不能确定者保留原样。"
                      "段内标点由词间停顿与连接词推定，仅供阅读参考，不代表讲者原意停顿。\n")

    n_para, nchar, n_cue = render(rows, dur, outdir, title, terms, "\n".join(meta_lines))

    json.dump({"audit": {k: v for k, v in rep.items() if k != "hallu"},
               "final": fin, "terms_hits": hits,
               "hallucinations": rep["hallu"], "breaks": breaks},
              open(os.path.join(outdir, "质检报告.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    loud.close()
    log(f"完成：{n_para} 段 / {nchar:,} 字 / {n_cue} 条字幕")
    log(f"  {outdir}/{title}_全文转录.md")
    log(f"  {outdir}/{title}_全文转录.txt")
    log(f"  {outdir}/{title}_字幕.srt")
    log(f"  {outdir}/质检报告.json")
    return 0
