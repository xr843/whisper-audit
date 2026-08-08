"""网页壳的纯函数部分。gradio 不在依赖里，故绝不在模块顶层 import 它。"""
import json

from whisper_audit.webui import build_cmd, collect_outputs, qa_summary


def test_build_cmd_uses_dash_m_form():
    """必须以 -m 形式起子进程——lessons 21：execv 重启只有 -m 路径是修过且锁死的。"""
    cmd = build_cmd("a.mp3", "out")
    assert cmd[1:4] == ["-m", "whisper_audit.cli", "run"]


def test_build_cmd_minimal():
    cmd = build_cmd("录音.mp3", "输出", engine="funasr", profile="fast")
    assert "--terms" not in cmd and "--diarize" not in cmd
    assert ["--engine", "funasr"] == cmd[cmd.index("--engine"):cmd.index("--engine") + 2]


def test_build_cmd_full():
    cmd = build_cmd("a.mp3", "o", terms="t.json", diarize=True, speakers=3)
    assert ["--terms", "t.json"] == cmd[cmd.index("--terms"):cmd.index("--terms") + 2]
    assert ["--speakers", "3"] == cmd[cmd.index("--speakers"):cmd.index("--speakers") + 2]


def test_speakers_without_diarize_not_passed():
    """UI 上只填人数不勾分离：不传 --speakers（CLI 端会警告并忽略，UI 端干脆不给）。"""
    cmd = build_cmd("a.mp3", "o", diarize=False, speakers=3)
    assert "--speakers" not in cmd and "--diarize" not in cmd


def test_collect_outputs_finds_all(tmp_path):
    for n in ("x_全文转录.md", "x_全文转录.txt", "x_字幕.srt", "质检报告.json"):
        (tmp_path / n).write_text("", encoding="utf-8")
    out = collect_outputs(str(tmp_path))
    assert all(out.values()), out


def test_collect_outputs_missing_dir():
    out = collect_outputs("/不存在的目录")
    assert not any(out.values())


def test_qa_summary_renders_healthy_report(tmp_path):
    p = tmp_path / "质检报告.json"
    p.write_text(json.dumps({
        "final": {"cover_pct": 96.0, "speech_pct": 93.9, "starved": 0},
        "hallucinations": [], "pinyin_fixes": [],
        "terms_hits": {"卫庄": 8}}, ensure_ascii=False), encoding="utf-8")
    s = qa_summary(str(p))
    assert "96.0%" in s and "卫庄×8" in s and "⚠" not in s


def test_qa_summary_warns_on_starved(tmp_path):
    p = tmp_path / "质检报告.json"
    p.write_text(json.dumps({
        "final": {"cover_pct": 90.0, "speech_pct": 80.0, "starved": 2},
        "hallucinations": []}, ensure_ascii=False), encoding="utf-8")
    assert "⚠" in qa_summary(str(p))


def test_qa_summary_broken_file_says_so(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    assert "失败" in qa_summary(str(p))


def test_localhost_exempted_from_proxy():
    """代理环境下 gradio 的 localhost 自检会被劫走，报错信息诱导 share=True
    （公网分享）。豁免 localhost 是唯一不违背「音频不出本机」的修法。"""
    from whisper_audit.webui import exempt_localhost_from_proxy
    env = {"NO_PROXY": "internal.corp", "no_proxy": ""}
    out = exempt_localhost_from_proxy(env)
    for k in ("no_proxy", "NO_PROXY"):
        assert "127.0.0.1" in out[k] and "localhost" in out[k]
    assert "internal.corp" in out["NO_PROXY"], "既有豁免不许被覆盖"


def test_supported_kwargs_filters_unknown():
    """gradio 4/5/6 参数各有增删，展示性参数宁可丢弃不能让 UI 起不来。"""
    from whisper_audit.webui import supported_kwargs

    def fn(a, b=1): ...
    assert supported_kwargs(fn, a=1, b=2, nope=3) == {"a": 1, "b": 2}
