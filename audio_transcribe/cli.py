"""命令行入口。"""
import argparse
import json
import os
import sys

from . import hms, log
from .audio import Loudness, ensure_cuda_libs, prepare_audio
from .audit import SPEECH_RATE, audit, audit_rows, find_breaks, in_any
from .engines.whisper import repatch, transcribe_pass
from .merge import combine
from .render import raw_text, render, terms_hits

# ---------------------------------------------------------------- 参数档

# 非 whisper 引擎的模型名是固定的，日志与出稿说明里直接报全称；
# whisper 走 --model/档位解析，所以不在这张表里（get 会落到 model_name）。
ENGINE_LABEL = {"funasr": "FunASR SeacoParaformer", "qwen": "Qwen3-ASR-0.6B"}

# batch=16 必须配 int8_float16：8GB 卡上 fp16+batch16 会 OOM（实测）。
# int8 本身不提速，它的价值就是省出显存来开大 batch —— 那才是提速来源。
# beam 默认 1（2026-08-06 改，原为 5）。四域实测 beam5 从未赢过像样的：
#   FLEURS 朗读   7.56 vs 7.57  平
#   演讲 ZH00004  7.10 vs 4.18  beam1 大胜，删除 685→243
#   慢速歌唱          90.8 vs 70.9  beam1 大胜，删除 2017→1360
#   讲课 ZH00005  8.35 vs 8.67  beam5 小胜 0.3pp——但该域应改用 --engine funasr
# beam search 在困难音频上更容易搜到「整段无语音」的路径整段放弃，
# 直接违背「不遗漏」的立项目标；还慢 12%。口音域没有直接数据，是剩余不确定性。
PROFILES = {
    # 单人讲授、音质好：单路 + 补转即可，最快
    "lecture": {"two_pass": False, "chunk_coarse": 30, "batch": 16, "beam": 1,
                "compute": "int8_float16"},
    # 多人问答、口音重、内容重要：双路交叉，最全
    "meeting": {"two_pass": True, "chunk_coarse": 30, "chunk_fine": 10, "batch": 16, "beam": 1,
                "compute": "int8_float16"},
    # 只求快。2026-08-06 起用 large-v3-turbo（解码 32 层→4 层）：
    # FLEURS 945 条实测质量代价 0.9pp（7.56%→8.46%），长音频吞吐 62.3x vs 24.5x。
    # 旧方案（large-v3 + beam=1）只快 12% 且质量损失没量化过——turbo 两头都赢。
    "fast": {"two_pass": False, "chunk_coarse": 30, "batch": 16, "beam": 5,
             "compute": "int8_float16",
             "model": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"},
}

# 计划里还有一个 accurate 档，**故意没有加**。
#
# 它本该是「双引擎 + 拼音纠错 + LLM 校订」，但眼下三样都不成立：
#   - 中文专用第二引擎因环境问题装不上（见 docs/measurements.md）
#   - 拼音纠错会改坏正确文本，且没有 CER 证据说它净收益为正
#   - LLM 校订要把正文发出本机，这个决定不该由「选了哪个档位」隐含替用户做出
#
# 去掉这三样，accurate 就和 meeting 一模一样。加一个名字叫「高精度」、
# 实则毫无差别的档位，比不加更糟——它会让人以为自己开了什么。
# 等金标 CER 出来、能证明哪几样真有收益，再按数据组这个档。


# ---------------------------------------------------------------- 主流程

def add_run_args(ap):
    ap.add_argument("audio")
    ap.add_argument("-o", "--outdir", default=None, help="输出目录，默认与音频同名")
    ap.add_argument("--profile", default="meeting", choices=list(PROFILES))
    ap.add_argument("--terms", default=None, help="术语修正表 json")
    ap.add_argument("--title", default=None)
    ap.add_argument("--keep-break", action="store_true", help="不剔除中场休息段")
    ap.add_argument("--engine", default="whisper", choices=["whisper", "funasr", "qwen"],
                    help="主引擎。funasr=SeacoParaformer：SpeechIO 演讲/讲课域实测 "
                         "CER 2.06%%/2.44%%（whisper 系 4.2~8.7%%）、删除少 6~10 倍、"
                         "速度更快——**标准普通话内容选它**。whisper 保留为默认：慢速歌唱等"
                         "困难域 funasr 会崩（见 docs/measurements.md），重口音未实测")
    ap.add_argument("--model", default=None,
                    help="faster-whisper 模型名。默认 large-v3；fast 档默认 turbo。"
                         "显式传入则覆盖档位设置")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--compute", default=None, help="覆盖档位里的 compute_type")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--pinyin-fix", action="store_true",
                    help="开启拼音级术语纠错。**默认关**——无调拼音会把两个真实存在、"
                         "含义不同的词判成同一个（节余/结余、空值/控制），"
                         "补多少术语都堵不住。用之前先读 docs/measurements.md")
    ap.add_argument("--loose-pinyin", action="store_true",
                    help="拼音纠错启用近音归并（zh/z、ang/an…）。"
                         "能多修几处，但词边界误伤风险更高，默认关")
    ap.add_argument("--diarize", action="store_true",
                    help="标注说话人（cam++ 声纹聚类）。只加标签，绝不改正文；"
                         "只认出一人时不标")
    ap.add_argument("--speakers", type=int, default=None,
                    help="已知说话人数，配合 --diarize；默认按声纹距离自动定")
    ap.add_argument("--polish", action="store_true",
                    help="出稿前过一遍 LLM 同音校订。默认关；正文会发往你指定的 endpoint")
    ap.add_argument("--polish-dry-run", action="store_true",
                    help="只打印将要发送的内容与块数，不发任何请求")
    ap.add_argument("--llm-base-url", default="https://api.openai.com/v1",
                    help="OpenAI 兼容 endpoint")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    return ap


def add_goldset_args(ap):
    ap.add_argument("outdir", help="跑完的输出目录")
    ap.add_argument("--from", dest="start", default=None,
                    help="起点，00:10:00 / 10:00 / 600 均可")
    ap.add_argument("--to", dest="end", default=None, help="终点，同上")
    ap.add_argument("-o", "--out", required=True, help="待校对稿输出路径 .gold.tsv")
    return ap


def add_eval_args(ap):
    ap.add_argument("--gold", default=None, help="人工校对过的 .gold.tsv")
    ap.add_argument("--hyp", default=None, help="输出目录，或直接给 .srt 文件")
    ap.add_argument("--manifest", default=None,
                    help="公开集 jsonl（每行 {\"audio\":…, \"text\":…}），"
                         "与 --gold/--hyp 二选一")
    ap.add_argument("--json", default=None, help="把报告写成 json")
    ap.add_argument("--engine", default="whisper", choices=["whisper", "funasr", "qwen"],
                    help="仅 --manifest 时用：拿哪个引擎跑公开集")
    ap.add_argument("--model", default="large-v3", help="仅 --manifest 时用")
    ap.add_argument("--device", default="cuda", help="仅 --manifest 时用")
    ap.add_argument("--language", default="zh", help="仅 --manifest 时用")
    ap.add_argument("--limit", type=int, default=None,
                    help="仅 --manifest 时用：只评前 N 条（抽样要在报告里写明）")
    return ap


# 已实现的子命令。加新子命令时必须同步这里，否则它会被当成音频文件名。
SUBCOMMANDS = ("run", "goldset", "eval")

_BUILDERS = {"run": add_run_args, "goldset": add_goldset_args, "eval": add_eval_args}
_HELP = {"run": "转录音频", "goldset": "从输出里切一段生成待校对稿",
         "eval": "拿校对好的金标算 CER"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # 不写子命令时默认走 run，保证 `python3 transcribe.py 录音.mp3` 行为不变。
    # 判据不能用 startswith("-")，否则 `-o 输出目录 录音.mp3` 这种写法会漏掉 run。
    if argv and argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "run")

    ap = argparse.ArgumentParser(prog="audio-transcribe",
                                 description="长音频转录流水线（以不遗漏为目标）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in SUBCOMMANDS:
        _BUILDERS[name](sub.add_parser(name, help=_HELP[name]))
    if not argv:
        ap.print_help()
        return 1
    args = ap.parse_args(argv)
    return {"run": cmd_run, "goldset": cmd_goldset, "eval": cmd_eval}[args.cmd](args)


def cmd_goldset(args):
    from .goldset import find_srt, make_goldset, parse_hms, write_tsv
    srt = args.outdir if args.outdir.endswith(".srt") else find_srt(args.outdir)
    rows = make_goldset(srt, parse_hms(args.start), parse_hms(args.end))
    if not rows:
        log("时间窗内没有字幕，检查 --from/--to")
        return 1
    write_tsv(rows, args.out)
    nchar = sum(len(t) for _, _, t in rows)
    log(f"待校对稿 {args.out}：{len(rows)} 行 / {nchar:,} 字"
        f"（{hms(rows[0][0])}–{hms(rows[-1][0])}）")
    log("  现在打开它，**只改错字**——不要动时间戳、不要动格式、不要合并行。")
    log(f"  改完跑：audio-transcribe eval --gold {args.out} --hyp {args.outdir}")
    return 0


def _eval_manifest(args):
    """跑公开集：逐条转录再按字加权汇总。

    每条都要过一遍 ASR，所以这条路慢得多，且需要装好 ASR 后端。
    """
    from bench.manifest import eval_manifest, read_manifest
    ensure_cuda_libs()
    items = read_manifest(args.manifest)
    if args.limit:
        items = items[:args.limit]
    log(f"公开集 {args.manifest}：{len(items)} 条，引擎 {args.engine}，开始逐条转录…")
    done = [0]

    def tick():
        done[0] += 1
        if done[0] % 100 == 0:
            log(f"  {done[0]}/{len(items)}")

    if args.engine != "whisper":
        from .engines import get_engine
        eng = get_engine(args.engine, device=args.device)

        def run(path):
            d = eng.transcribe(path)
            tick()
            return "".join(s["text"] for s in d["segments"])
    else:
        # 模型只加载一次。走 transcribe_pass 会每条重建一次模型——
        # AISHELL test 有 7,000 多条，那是几十小时的纯加载时间。
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        model = WhisperModel(args.model, device=args.device,
                             compute_type="int8_float16")
        pipe = BatchedInferencePipeline(model=model)

        def run(path):
            segs, _ = pipe.transcribe(path, language=args.language, batch_size=16,
                                      beam_size=5, condition_on_previous_text=False)
            tick()
            return "".join(s.text for s in segs)

    return eval_manifest(items, run)


def cmd_eval(args):
    from .goldset import evaluate_srt, find_srt, format_report
    if args.manifest:
        rep = _eval_manifest(args)
    elif args.gold and args.hyp:
        srt = args.hyp if args.hyp.endswith(".srt") else find_srt(args.hyp)
        rep = evaluate_srt(args.gold, srt)
    else:
        log("需要 --gold + --hyp，或者 --manifest")
        return 2
    print(format_report(rep))
    if args.json:
        json.dump(rep, open(args.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        log(f"报告已写入 {args.json}")
    return 0


def run_polish(rows, terms, args):
    """LLM 同音校订。逐行处理以保住时间戳与词级时间戳。

    两道锁，缺一不可：

    1. `constrain` 逐字校验拼音——只放行同音替换，长度变了整块拒绝。
    2. 行数校验——用换行拼接、返回后核对行数。`constrain` 只保长度不保结构，
       LLM 把两行并成一行、长度恰好不变的情况它看不出来，行数能。

    **这两道锁保证的是「不增、不删、不换位」，不保证「不改义」。**
    中文里同音替换恰恰是改变含义最常见的方式，实测这些都能通过护栏：

        账上还有一万元      → 亿万元    金额差一万倍
        公司有权力处分      → 权利      法律含义相反
        合同约定定金五万元  → 订金      可否退还相反
        收到捐赠            → 受到

    所以这一档的改动必须逐条核对，不能因为「有护栏」就当它安全。

    校订之后的拼音纠错兜底**只在用户确实开了 --pinyin-fix 时才跑**——
    它要覆盖的是「LLM 把已纠正的词改回去」，没开就没有可覆盖的对象，
    硬跑等于替用户做了他没同意的改动。
    """
    import os as _os

    from .polish import polish
    from .terms import pinyin_fix

    key = _os.environ.get("AUDIO_TRANSCRIBE_LLM_KEY", "")
    if not key and not args.polish_dry_run:
        raise SystemExit("需要环境变量 AUDIO_TRANSCRIBE_LLM_KEY（不要写进命令行，会落进 shell 历史）")

    joined = "\n".join(r["text"] for r in rows)
    log(f"LLM 同音校订：{len(rows)} 行 / {len(joined):,} 字 → {args.llm_base_url}"
        + ("　[dry-run，不发请求]" if args.polish_dry_run else ""))
    fixed, rep = polish(joined, base_url=args.llm_base_url, model=args.llm_model,
                        api_key=key, dry_run=args.polish_dry_run)

    if args.polish_dry_run:
        # dry-run 承诺的是「只打印将要发送什么」。它**一个字都不能改**——
        # 曾经这里无条件跑兜底纠错，把「节余」改成「结余」还不记账。
        log(f"  {rep['chunks']} 块（dry-run，正文未改动）")
        return rep, []

    parts = fixed.split("\n")
    if len(parts) == len(rows):
        for r, t in zip(rows, parts):
            r["text"] = t
    else:
        log(f"  ⚠ 返回 {len(parts)} 行 ≠ 原 {len(rows)} 行，结构被破坏，本次校订整体作废")
        rep["structure_rejected"] = True

    # 兜底只在用户确实开了拼音纠错时才有意义——它要「覆盖 LLM 的反悔」，
    # 没开就根本没有可覆盖的东西，硬跑等于替用户做了他没同意的改动。
    hits = []
    if args.pinyin_fix:
        for r in rows:
            r["text"], h = pinyin_fix(r["text"], terms, loose=args.loose_pinyin)
            hits.extend({**x, "t": round(r["start"], 2), "stage": "polish兜底"} for x in h)

    log(f"  {rep['chunks']} 块：接受 {rep['accepted']} 处、拒绝 {rep['rejected']} 处、"
        f"整块拒绝 {rep['length_rejected']} 块、请求失败 {rep['failed']} 块")
    return rep, hits


def cmd_run(args):
    # 提前校验：真实录音要跑几十分钟，不能等转录全做完才告诉用户缺环境变量。
    if args.polish and not args.polish_dry_run \
            and not os.environ.get("AUDIO_TRANSCRIBE_LLM_KEY"):
        log("--polish 需要环境变量 AUDIO_TRANSCRIBE_LLM_KEY"
            "（不要写进命令行，会落进 shell 历史）")
        return 2

    ensure_cuda_libs()

    src = os.path.abspath(args.audio)
    title = args.title or os.path.splitext(os.path.basename(src))[0]
    outdir = os.path.abspath(args.outdir or os.path.join(os.path.dirname(src), title + "_转录"))
    work = os.path.join(outdir, ".work")
    os.makedirs(work, exist_ok=True)

    terms = json.load(open(args.terms, encoding="utf-8")) if args.terms else {}
    cfg = PROFILES[args.profile]
    compute = args.compute or (cfg["compute"] if args.device != "cpu" else "int8")
    # 模型解析：显式 --model > 档位指定 > large-v3
    model_name = args.model or cfg.get("model", "large-v3")
    mk = dict(model_name=model_name, device=args.device, language=args.language)
    log(f"档位 {args.profile}　引擎 {args.engine}　模型 {ENGINE_LABEL.get(args.engine, model_name)}／{args.device}／{compute}　输出 {outdir}")

    wav = prepare_audio(src, work)
    loud = Loudness(wav)

    if args.engine != "whisper":
        # 单路：不同 chunk 的双路是 whisper 特有的权衡，对非自回归引擎无意义。
        # 补转仍走 whisper——主引擎漏掉的区段拿另一个引擎复核，正好跨引擎互补。
        p1 = os.path.join(work, "pass1.json")
        if os.path.exists(p1):
            log(f"复用已有 {os.path.basename(p1)}")
            passes = [json.load(open(p1, encoding="utf-8"))]
        else:
            from .engines import get_engine
            log(f"转录 engine={args.engine}…")
            d = get_engine(args.engine, device=args.device).transcribe(wav)
            json.dump(d, open(p1, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            log(f"  完成 {len(d['segments'])} 段 / "
                f"{sum(len(s['text']) for s in d['segments']):,} 字")
            passes = [d]
        if cfg["two_pass"]:
            log(f"  {args.engine} 引擎为单路，忽略档位的 two_pass")
    else:
        passes = [transcribe_pass(wav, os.path.join(work, "pass1.json"),
                                  cfg["chunk_coarse"], cfg["batch"], cfg["beam"],
                                  compute, **mk)]
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
    rows, pyhits = combine(passes, patch, terms, breaks, dur, rep.get("drop", []),
                           pinyin=args.pinyin_fix,
                           loose=args.loose_pinyin, return_hits=True)
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

    n_spk = 0
    if args.diarize:
        # 放在最后：说话人标注只读 rows 的时间、只写 speaker 字段，
        # 跑在正文全部定稿之后，出问题也影响不到一个字。
        # 整段 try 包住是刻意的——真实录音跑一小时才走到这里，绝不能
        # 因为「可选标注」的模型没下载/显存不够，把已经转好的稿子搞没。
        try:
            from .diarize import assign_speakers
            assign_speakers(rows, wav, n_speakers=args.speakers, device=args.device)
            seen = [r.get("speaker") for r in rows]
            n_spk = len({s for s in seen if s})
            log(f"说话人分离：{n_spk} 人"
                + (f"，{sum(1 for s in seen if not s)} 段未定" if not all(seen) else "")
                + ("　（只认出一人，出稿不加标签）" if n_spk < 2 else ""))
        except Exception as e:
            log(f"  ⚠ 说话人分离失败（{type(e).__name__}: {e}），跳过，正文不受影响")

    prep = None
    if args.polish or args.polish_dry_run:
        prep, polish_hits = run_polish(rows, terms, args)
        pyhits = pyhits + polish_hits      # 兜底阶段的改动也要进同一本账

    if pyhits:
        from collections import Counter
        c = Counter((h["from"], h["to"]) for h in pyhits)
        top = "、".join(f"{a}→{b}×{n}" for (a, b), n in c.most_common(3))
        log(f"拼音纠错：{len(pyhits)} 处 / {len(c)} 种（{top}）"
            + ("　近音归并已开启" if args.loose_pinyin else ""))

    # 必须在替换**之前**数：正文里源词早被换掉了，事后统计只会全是 0
    hits = terms_hits(raw_text(passes), terms)
    if hits:
        miss = [k for k, v in hits.items() if v == 0]
        log(f"术语表：{len(hits)} 条，命中 {sum(1 for v in hits.values() if v)} 条"
            + (f"，0 命中 {len(miss)} 条（{'、'.join(miss[:5])}{'…' if len(miss)>5 else ''}）"
               if miss else ""))

    meta_lines = [
        f"**转录方式**　{ENGINE_LABEL.get(args.engine) or "faster-whisper " + model_name}，{len(passes)} 路 + "
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
    if prep:
        meta_lines.append(
            f"**LLM 同音校订**　已启用，接受 {prep['accepted']} 处、拒绝 {prep['rejected']} 处。"
            "护栏只允许同音替换（读音不同的改动一律还原），因此不会增删内容；"
            "但**同音替换本身可能改变含义**（如 权力/权利、定金/订金、一/亿），"
            "这一档的改动请对照质检报告逐条核对。\n")
    if n_spk >= 2:
        meta_lines.append(
            f"**说话人**　按声纹自动聚类为 {n_spk} 人，标为 S1…S{n_spk}（按出场顺序）。"
            "标签是推定的，且短促插话容易归错人；标签只是标注，未改动任何正文文字。\n")
    meta_lines.append("**注意**　语音识别对专有名词与专业术语存在同音误识，"
                      f"已按术语表统一校正可确定者（其中拼音级纠错 {len(pyhits)} 处，"
                      "逐条列在质检报告的 pinyin_fixes 里），不能确定者保留原样。"
                      "段内标点由词间停顿与连接词推定，仅供阅读参考，不代表讲者原意停顿。\n")

    n_para, nchar, n_cue = render(rows, dur, outdir, title, terms, "\n".join(meta_lines))

    json.dump({"audit": {k: v for k, v in rep.items() if k != "hallu"},
               "final": fin, "terms_hits": hits, "pinyin_fixes": pyhits,
               "polish": prep,
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
