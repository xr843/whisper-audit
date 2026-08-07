"""命令行分发的回归测试。

子命令化之后最容易静默坏掉的是「不写子命令时默认走 run」这条路径——
它坏了不会报错，只会让 `python3 transcribe.py 录音.mp3` 变成 usage 提示。
"""
import argparse

import pytest

from whisper_audit import cli


def parse(case):
    """复刻 main() 的分发逻辑，但不执行转录。"""
    argv = list(case)
    if argv and argv[0] not in cli.SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "run")
    ap = argparse.ArgumentParser(prog="whisper-audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cli.add_run_args(sub.add_parser("run"))
    return ap.parse_args(argv)


def test_bare_audio_path_defaults_to_run():
    """`python3 transcribe.py 录音.mp3` 必须继续可用。"""
    a = parse(["录音.mp3"])
    assert a.cmd == "run" and a.audio == "录音.mp3"


def test_leading_option_still_defaults_to_run():
    """判据不能写成 startswith('-')，否则选项在前的写法会漏掉 run。"""
    a = parse(["-o", "输出", "录音.mp3"])
    assert a.cmd == "run" and a.audio == "录音.mp3" and a.outdir == "输出"


def test_explicit_run_subcommand_works():
    a = parse(["run", "录音.mp3"])
    assert a.cmd == "run" and a.audio == "录音.mp3"


def test_long_option_before_positional():
    a = parse(["--profile", "fast", "录音.mp3"])
    assert a.audio == "录音.mp3" and a.profile == "fast"


def test_help_is_not_swallowed_by_run():
    """--help 要给顶层用法，不能被塞进 run 子命令。"""
    with pytest.raises(SystemExit) as e:
        parse(["--help"])
    assert e.value.code == 0


def test_empty_argv_prints_help_instead_of_crashing():
    assert cli.main([]) == 1


def test_every_declared_subcommand_has_a_builder():
    """SUBCOMMANDS 漏登记会让子命令被当成音频文件名——静默且难查。

    必须拿 main() 真正用的 _BUILDERS 来比，不能在测试里另建一份注册表，
    否则加了子命令后这条测试只会红在自己身上，抓不到真正的漂移。
    """
    assert set(cli.SUBCOMMANDS) == set(cli._BUILDERS)
    assert set(cli.SUBCOMMANDS) == set(cli._HELP)


def test_all_subcommands_parse_their_own_args():
    ap = argparse.ArgumentParser(prog="whisper-audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in cli.SUBCOMMANDS:
        cli._BUILDERS[name](sub.add_parser(name))
    assert ap.parse_args(["goldset", "outdir", "-o", "g.tsv"]).out == "g.tsv"
    assert ap.parse_args(["eval", "--gold", "g.tsv", "--hyp", "d"]).gold == "g.tsv"


def test_subcommand_name_is_not_swallowed_as_audio_path():
    """goldset/eval 必须被识别为子命令，而不是被当成音频文件名塞给 run。"""
    for name in ("goldset", "eval"):
        argv = [name]
        if argv[0] not in cli.SUBCOMMANDS:
            argv.insert(0, "run")
        assert argv[0] == name


def test_eval_requires_gold_and_hyp_or_manifest():
    """两种入口二选一；都不给必须报错退出，而不是崩在 None 上。"""
    assert cli.main(["eval"]) == 2
    assert cli.main(["eval", "--gold", "g.tsv"]) == 2


def test_eval_accepts_manifest_form():
    ap = argparse.ArgumentParser(prog="whisper-audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in cli.SUBCOMMANDS:
        cli._BUILDERS[name](sub.add_parser(name))
    a = ap.parse_args(["eval", "--manifest", "bench/aishell_test.jsonl"])
    assert a.manifest == "bench/aishell_test.jsonl"
    assert a.gold is None and a.hyp is None


def test_polish_without_key_fails_before_transcribing(monkeypatch, tmp_path):
    """真实录音要跑几十分钟，缺环境变量必须在开跑前就报，不能等转完才说。"""
    monkeypatch.delenv("WHISPER_AUDIT_LLM_KEY", raising=False)
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"not really audio")
    assert cli.main(["run", str(audio), "--polish"]) == 2
    assert not (tmp_path / "x_转录").exists(), "报错前不该建输出目录"


def test_polish_dry_run_does_not_need_a_key(monkeypatch):
    """dry-run 不发请求，不该要 key。"""
    monkeypatch.delenv("WHISPER_AUDIT_LLM_KEY", raising=False)
    ap = argparse.ArgumentParser(prog="whisper-audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in cli.SUBCOMMANDS:
        cli._BUILDERS[name](sub.add_parser(name))
    a = ap.parse_args(["run", "x.wav", "--polish-dry-run"])
    assert a.polish_dry_run and not a.polish


def test_fast_profile_selects_turbo_model():
    """fast 档 2026-08-06 起用 turbo：质量代价已量化（FLEURS 0.9pp），
    长音频吞吐 62.3x vs 24.5x。旧方案 beam=1 只快 12% 且损失从未量化。"""
    cfg = cli.PROFILES["fast"]
    assert "turbo" in cfg["model"]
    assert cfg["beam"] == 5, "turbo 的速度来自减层，不必再拿 beam=1 牺牲质量"


def test_explicit_model_overrides_profile(monkeypatch):
    """显式 --model 必须压过档位里的 model 字段。"""
    ap = argparse.ArgumentParser(prog="whisper-audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in cli.SUBCOMMANDS:
        cli._BUILDERS[name](sub.add_parser(name))
    a = ap.parse_args(["run", "x.wav", "--profile", "fast", "--model", "large-v3"])
    cfg = cli.PROFILES[a.profile]
    assert (a.model or cfg.get("model", "large-v3")) == "large-v3"
    b = ap.parse_args(["run", "x.wav", "--profile", "fast"])
    assert (b.model or cfg.get("model", "large-v3")).endswith("turbo")


def test_default_profiles_keep_large_v3():
    """质量榜首是 large-v3（FLEURS 7.56% vs turbo/paraformer 8.45~8.46%）——
    默认档不许被悄悄换成快模型。"""
    for name in ("lecture", "meeting"):
        assert "model" not in cli.PROFILES[name]


def test_default_profiles_keep_beam_1():
    """beam 1 是四域实测的结论，不是随手填的，必须锁住。

    2026-08-06 实测 beam5 从未像样地赢过：
        FLEURS 朗读   7.56 vs 7.57  平
        演讲 ZH00004  7.10 vs 4.18  beam1 大胜，删除 685→243
        慢速歌唱      90.8 vs 70.9  beam1 大胜，删除 2017→1360
        讲课 ZH00005  8.35 vs 8.67  beam5 小胜 0.3pp（该域应改用 funasr）

    beam search 在困难音频上更容易搜到「整段无语音」的路径整段放弃，
    直接违背「不遗漏」的立项目标；还慢 12%。
    fast 档是例外（turbo 减层已经够快，见 test_fast_profile_selects_turbo_model）。
    """
    for name in ("lecture", "meeting"):
        assert cli.PROFILES[name]["beam"] == 1, f"{name} 档的 beam 被改动了"


def test_meeting_is_two_pass_and_lecture_is_not():
    """双路交叉是 meeting 档的全部含义，改掉它这个档就没有存在理由了。"""
    assert cli.PROFILES["meeting"]["two_pass"] is True
    assert cli.PROFILES["lecture"]["two_pass"] is False
    assert cli.PROFILES["fast"]["two_pass"] is False


def test_default_profile_is_single_pass():
    """默认档 2026-08-07 从 meeting 改为 lecture，依据是目标域消融实测。

    SpeechIO ZH00004，同一份参考，两种音频结构都复现：

        配置                        有间隔 73.8min   连续 65.5min
        裸引擎单路                     4.15%             —
        lecture（单路+审计+补转）      4.15%           4.29%
        meeting（双路合并）            9.69%           9.87%

    双路合并的两个输入是 4.15% 和 4.77%，合出来 9.69%——**比两个输入都差**。
    合并永远不该输给它的任一输入，所以这是 combine() 的缺陷，不是语料问题。

    这条锁的是「默认档必须是单路」，不是锁死具体档名——将来档位重组时，
    只要默认仍是单路即可。
    """
    a = parse(["录音.mp3"])
    assert cli.PROFILES[a.profile]["two_pass"] is False, \
        f"默认档 {a.profile} 是双路——干净普通话上实测 CER 会翻 2.3 倍"


def test_batch_requires_int8_float16():
    """batch=16 必须配 int8_float16：8GB 卡上 fp16+batch16 会 OOM（实测）。

    int8 本身不提速，它的价值就是省出显存来开大 batch——那才是提速来源。
    两者必须成对出现，改单边会直接 OOM 或白白慢下来。
    """
    for name, cfg in cli.PROFILES.items():
        if cfg.get("batch", 1) > 8:
            assert cfg["compute"] == "int8_float16", f"{name} 档 batch 大但没配 int8_float16"


def test_run_engine_flag_exists_with_whisper_default():
    """funasr 在标准普通话域碾压（SpeechIO 2.06%/2.44% vs whisper 4.2~8.7%），
    但慢速歌唱域会崩、重口音未测——所以是显式选项，whisper 仍是默认。"""
    ap = argparse.ArgumentParser(prog="whisper-audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in cli.SUBCOMMANDS:
        cli._BUILDERS[name](sub.add_parser(name))
    a = ap.parse_args(["run", "x.wav"])
    assert a.engine == "whisper"
    b = ap.parse_args(["run", "x.wav", "--engine", "funasr"])
    assert b.engine == "funasr"
