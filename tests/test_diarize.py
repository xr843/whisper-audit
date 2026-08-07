"""说话人分离测试。

全部纯 CPU、秒级跑完：`cluster_speakers` 用人造声纹向量测聚类算法本身；
`assign_speakers` 通过 `embed_fn` 注入假的声纹抽取函数测编排逻辑（短段继承、
标签命名、text 保真）。真实 cam++ 需要 GPU + funasr（CI 没装），一律不在
单测里调用——同 tests/test_funasr_adapter.py 对 FunASREngine 的处理方式：
纯函数单测到位，真机验证走 diarize-report.md。
"""
import json
import pathlib

import numpy as np
import pytest

from whisperaudit import diarize as D
from whisperaudit.diarize import (
    _label_names,
    _slice_audio,
    assign_speakers,
    cluster_speakers,
)


# ---------------------------------------------------------------- 造数据的小工具

def _one_speaker(n=8, dim=192, seed=1, spread=0.02):
    """单一说话人：一个中心向量 + 很小的噪声，两两余弦距离应远低于任何合理阈值。"""
    rng = np.random.default_rng(seed)
    center = rng.normal(size=dim)
    return center + rng.normal(scale=spread, size=(n, dim))


def _two_speakers(n_a=5, n_b=4, dim=192, seed=0, spread=0.01):
    """两个方向近乎相反（余弦距离趋近 2，聚类问题里最容易分开的情形）的簇。"""
    rng = np.random.default_rng(seed)
    a_center = rng.normal(size=dim)
    b_center = -a_center
    a = a_center + rng.normal(scale=spread, size=(n_a, dim))
    b = b_center + rng.normal(scale=spread, size=(n_b, dim))
    return np.vstack([a, b]), n_a, n_b


@pytest.fixture
def wav16k(tmp_path):
    """10 秒静音 16k 单声道 wav——assign_speakers 需要真实存在的文件才能切片，
    但测试里声纹由注入的假 embed_fn 决定，音频内容本身无所谓。"""
    import soundfile as sf
    p = tmp_path / "audio16k.wav"
    sf.write(str(p), np.zeros(16000 * 10, dtype="float32"), 16000)
    return str(p)


# ---------------------------------------------------------------- cluster_speakers：纯聚类算法

def test_two_clearly_separated_speakers_are_split():
    embeds, n_a, n_b = _two_speakers()
    labels = cluster_speakers(embeds)
    labels_a, labels_b = list(labels[:n_a]), list(labels[n_a:])
    assert len(set(labels_a)) == 1, "同一说话人的段不该被切成多类"
    assert len(set(labels_b)) == 1
    assert labels_a[0] != labels_b[0], "方向近乎相反的两簇必须被分开"


def test_single_speaker_is_not_split():
    embeds = _one_speaker(n=10)
    labels = cluster_speakers(embeds)
    assert len(set(labels)) == 1


def test_n_speakers_forces_exact_count():
    """不管数据本身长什么样，指定 n_speakers=3 必须严格给 3 类。"""
    rng = np.random.default_rng(3)
    embeds = rng.normal(size=(6, 192))
    labels = cluster_speakers(embeds, n_speakers=3)
    assert len(set(labels)) == 3


def test_auto_detect_finds_two_for_obvious_two_clusters():
    embeds, _, _ = _two_speakers()
    labels = cluster_speakers(embeds, n_speakers=None)
    assert len(set(labels)) == 2


def test_empty_input_does_not_crash():
    labels = cluster_speakers(np.zeros((0, 192)))
    assert len(labels) == 0


def test_single_embedding_does_not_crash():
    labels = cluster_speakers(np.ones((1, 192)))
    assert list(labels) == [0]


def test_n_speakers_larger_than_samples_is_clamped():
    """只有 3 条声纹却要 10 类：截到样本数，不报错、不产生空类。"""
    embeds = _one_speaker(n=3)
    labels = cluster_speakers(embeds, n_speakers=10)
    assert len(set(labels)) == 3


def test_n_speakers_zero_or_negative_raises():
    with pytest.raises(ValueError):
        cluster_speakers(_one_speaker(n=3), n_speakers=0)


def test_cluster_labels_are_magnitude_invariant():
    """余弦距离只看方向：整体放大 100 倍，聚类结果不该变——防止误用欧氏距离。"""
    embeds, _, _ = _two_speakers()
    labels1 = list(cluster_speakers(embeds))
    labels2 = list(cluster_speakers(embeds * 100))
    assert labels1 == labels2


# ---------------------------------------------------------------- 内部小函数

def test_label_names_orders_by_first_appearance():
    """S1/S2 按首次出现顺序编号，不是簇 id 本身——保证「S1 是先开口的人」。"""
    assert _label_names([2, 2, 0, 0, 2]) == ["S1", "S1", "S2", "S2", "S1"]


def test_slice_audio_clamps_to_bounds():
    audio = np.arange(10, dtype="float32")
    out = _slice_audio(audio, sr=1, start=5, end=100)
    assert len(out) == 5          # end 超界：截到音频末尾，不报错
    out2 = _slice_audio(audio, sr=1, start=5, end=5)
    assert len(out2) >= 1          # 零长度区间：返回占位，不崩


# ---------------------------------------------------------------- assign_speakers：编排逻辑

def test_assign_speakers_splits_two_distinct_speakers(wav16k):
    rows = [
        {"start": 0.0, "end": 2.0, "text": "甲说话一"},
        {"start": 2.0, "end": 4.0, "text": "甲说话二"},
        {"start": 4.0, "end": 6.0, "text": "乙说话一"},
        {"start": 6.0, "end": 8.0, "text": "乙说话二"},
    ]

    def fn(clips, sr, device):
        out = []
        for i in range(len(clips)):
            v = np.zeros(192)
            v[0 if i < 2 else 1] = 1.0
            out.append(v)
        return np.array(out)

    out = assign_speakers(rows, wav16k, embed_fn=fn)
    assert out[0]["speaker"] == out[1]["speaker"] == "S1"
    assert out[2]["speaker"] == out[3]["speaker"] == "S2"
    assert out[0]["speaker"] != out[2]["speaker"]


def test_assign_speakers_keeps_single_speaker_together(wav16k):
    rows = [{"start": i * 1.0, "end": i * 1.0 + 0.8, "text": f"第{i}句"} for i in range(5)]

    def fn(clips, sr, device):
        rng = np.random.default_rng(7)
        center = rng.normal(size=192)
        return center + rng.normal(scale=0.01, size=(len(clips), 192))

    out = assign_speakers(rows, wav16k, embed_fn=fn)
    assert len({r["speaker"] for r in out}) == 1


def test_assign_speakers_respects_n_speakers(wav16k):
    rows = [{"start": i * 1.0, "end": i * 1.0 + 0.8, "text": f"第{i}段"} for i in range(6)]

    def fn(clips, sr, device):
        return np.eye(192)[:len(clips)]   # 6 个两两正交的向量，任意两条都"看起来不同"

    out = assign_speakers(rows, wav16k, n_speakers=2, embed_fn=fn)
    assert len({r["speaker"] for r in out}) == 2


def test_short_segments_inherit_previous_speaker_label(wav16k):
    rows = [
        {"start": 0.0, "end": 2.0, "text": "长段甲"},      # 可靠，声纹 A
        {"start": 2.0, "end": 2.2, "text": "短插话"},       # 不可靠，0.2s < min_dur 0.5s
        {"start": 2.2, "end": 4.2, "text": "长段乙"},       # 可靠，声纹 B——故意与 A 不同
    ]

    def fn(clips, sr, device):
        # 两条可靠段声纹明显不同（正交向量），分别应判成 S1/S2；
        # 这样短段到底"继承前一条"还是"继承后一条"才有区分度，不会两边都对。
        a = np.zeros(192)
        a[0] = 1.0
        b = np.zeros(192)
        b[1] = 1.0
        return np.array([a, b])

    out = assign_speakers(rows, wav16k, embed_fn=fn, min_dur=0.5)
    assert out[0]["speaker"] == "S1"
    assert out[1]["speaker"] == "S1", "短段应继承前一条可靠段的标签，不是后一条"
    assert out[2]["speaker"] == "S2"


def test_leading_short_segment_inherits_from_next_reliable_segment(wav16k):
    rows = [
        {"start": 0.0, "end": 0.2, "text": "开头短插话"},   # 前面没有可靠段可继承
        {"start": 0.2, "end": 2.2, "text": "长段"},
    ]

    def fn(clips, sr, device):
        return np.ones((len(clips), 192))

    out = assign_speakers(rows, wav16k, embed_fn=fn, min_dur=0.5)
    assert out[0]["speaker"] == "S1", "开头的短段找不到更早的锚点时，向后兜底"
    assert out[1]["speaker"] == "S1"


def test_all_segments_too_short_get_none(wav16k):
    rows = [
        {"start": 0.0, "end": 0.1, "text": "短"},
        {"start": 0.2, "end": 0.4, "text": "更短"},
    ]

    def fn(clips, sr, device):
        raise AssertionError("没有任何可靠段时不该调用 embed_fn")

    out = assign_speakers(rows, wav16k, embed_fn=fn, min_dur=0.5)
    assert out[0]["speaker"] is None
    assert out[1]["speaker"] is None


def test_text_field_is_never_modified(wav16k):
    """项目红线：说话人标签是新增字段，text 一字不能改。"""
    rows = [
        {"start": 0.0, "end": 2.0, "text": "第一段文字，逗号、顿号都在。"},
        {"start": 2.0, "end": 2.3, "text": "嗯"},
        {"start": 2.3, "end": 5.0, "text": "第三段完全不同的内容！？"},
    ]
    original_texts = [r["text"] for r in rows]

    def fn(clips, sr, device):
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(clips), 192))

    out = assign_speakers(rows, wav16k, embed_fn=fn, min_dur=0.5)
    assert [r["text"] for r in out] == original_texts
    assert len(out) == len(rows)


def test_assign_speakers_mutates_in_place(wav16k):
    rows = [{"start": 0.0, "end": 2.0, "text": "x"}]

    def fn(clips, sr, device):
        return np.ones((len(clips), 192))

    out = assign_speakers(rows, wav16k, embed_fn=fn)
    assert out is rows
    assert rows[0]["speaker"] == "S1"


def test_assign_speakers_empty_rows_does_not_crash():
    assert assign_speakers([], "不存在的文件.wav") == []


def test_assign_speakers_single_row_does_not_crash(wav16k):
    rows = [{"start": 0.0, "end": 2.0, "text": "只有一条"}]

    def fn(clips, sr, device):
        return np.ones((len(clips), 192))

    out = assign_speakers(rows, wav16k, embed_fn=fn)
    assert out[0]["speaker"] == "S1"


def test_row_order_is_preserved_regardless_of_input_order(wav16k):
    """rows 不按时间顺序传入时，输出顺序仍与输入一致（继承逻辑内部按时间排，
    但绝不能顺手把转录结果的行序改掉）。"""
    rows = [
        {"start": 5.0, "end": 6.0, "text": "乙"},
        {"start": 0.0, "end": 1.0, "text": "甲"},
    ]

    def fn(clips, sr, device):
        return np.eye(192)[:len(clips)]

    out = assign_speakers(rows, wav16k, embed_fn=fn, min_dur=0.5)
    assert [r["text"] for r in out] == ["乙", "甲"]


# ---------------------------------------------------------------- 真机声纹回归

_FIX = pathlib.Path(__file__).parent / "fixtures"


def _real_embeddings():
    import numpy as np
    return (np.load(_FIX / "spk_embeddings_2spk.npy"),
            json.loads((_FIX / "spk_truth_2spk.json").read_text(encoding="utf-8")))


def test_two_real_speakers_are_separated_at_default_threshold():
    """真机声纹回归：两位不同讲者各 12 段（正常语速），默认阈值必须分得开。

    这条测试的由来：阈值第一版按「慢速歌唱 + 讲解交替」的单人录音标定成 0.95，
    而那个域项目自己声明不支持。拿域外音频标定的结果是——两个真人被并成一簇，
    说话人分离功能直接失效。夹具是那 24 段的真实 cam++ 声纹（192 维），
    纯 CPU 可跑，不需要模型和 GPU。
    """
    E, truth = _real_embeddings()
    labels = D.cluster_speakers(E)
    assert len(set(labels)) == 2, f"两个真人应当分成 2 簇，实得 {len(set(labels))}"
    # 簇标签与真值一一对应（不关心哪个簇叫什么）
    pairs = set(zip(labels, truth))
    assert len(pairs) == 2, f"簇与说话人不是一一对应：{sorted(pairs)}"


def test_real_speaker_separation_has_margin_not_knife_edge():
    """0.40~0.80 都该全对——阈值不是走钢丝调出来的，有实测余量。"""
    E, truth = _real_embeddings()
    for th in (0.4, 0.5, 0.6, 0.7, 0.8):
        labels = D.cluster_speakers(E, threshold=th)
        assert len(set(labels)) == 2, f"阈值 {th} 下分成了 {len(set(labels))} 簇"


def test_explicit_speaker_count_overrides_threshold_on_real_embeddings():
    """自动定数不理想时，n_speakers 是逃生舱——真机声纹上必须严格生效。"""
    E, _ = _real_embeddings()
    for n in (1, 2, 3):
        assert len(set(D.cluster_speakers(E, n_speakers=n))) == n
