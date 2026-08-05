"""命令行分发的回归测试。

子命令化之后最容易静默坏掉的是「不写子命令时默认走 run」这条路径——
它坏了不会报错，只会让 `python3 transcribe.py 录音.mp3` 变成 usage 提示。
"""
import argparse

import pytest

from audio_transcribe import cli


def parse(case):
    """复刻 main() 的分发逻辑，但不执行转录。"""
    argv = list(case)
    if argv and argv[0] not in cli.SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "run")
    ap = argparse.ArgumentParser(prog="audio-transcribe")
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
    ap = argparse.ArgumentParser(prog="audio-transcribe")
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
    ap = argparse.ArgumentParser(prog="audio-transcribe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in cli.SUBCOMMANDS:
        cli._BUILDERS[name](sub.add_parser(name))
    a = ap.parse_args(["eval", "--manifest", "bench/aishell_test.jsonl"])
    assert a.manifest == "bench/aishell_test.jsonl"
    assert a.gold is None and a.hyp is None
