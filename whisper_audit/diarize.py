"""说话人分离：与 ASR 引擎解耦的独立三步——切片、抽 cam++ 声纹、聚类。

**不走 FunASR 的集成管线**（`AutoModel(..., vad_model=..., punc_model=..., spk_model=...)`）：
实测它会连带强制拉一个 1.05GB 的标点模型（`punc_ct-transformer_cn-en-common-
vocab471067-large`），而且该模型在 ModelScope 上缺 `jieba.c.dict`（404），
三次真机验证都在这一步阻塞（记录见 scratchpad/probe_spk.log）。改成这里的独立
三步，对 whisper/funasr/qwen 三个引擎都通用，不依赖任何一个引擎的内部管线：

    1. 对每条已有 transcript segment，从整段 wav 里切出对应音频（`_slice_audio`）
    2. 用 cam++（`speech_campplus_sv_zh-cn_16k-common`）抽 192 维声纹向量
    3. 聚类（凝聚层次聚类，余弦距离，average linkage）→ 给每条 segment 打标签

真机实测记录（2026-08-06，`speech_campplus_sv_zh-cn_16k-common`，本地已缓存，
无需授权 token，加载约 1 秒）：

- `model.generate(input=...)` 直接吃 16kHz float32 numpy array，也接受
  array 的 list 做一次性批量推理（顺序与输入一致，逐条 vs 批量数值一致，
  已用真实音频交叉验证）——不需要写临时 wav 文件。
- 返回的 `spk_embedding` 是 **CUDA tensor**，不是 numpy：直接 `np.asarray()`
  会报 `can't convert cuda:0 device type tensor to numpy`，必须先
  `.detach().cpu().numpy()`。
**阈值校准（两轮，第一轮标定错了域，记录在案）**

- 第一轮用一段**慢速歌唱与讲解交替**的单人录音标定：同人两两余弦距离
  中位 0.70、最大 1.02，要 0.95 才不把这一个人拆开。但那个域
  **项目自己声明不支持**（README 已知局限），拿域外音频标定，把域内标坏了。
- 第二轮改用**域内**真实双说话人音频（两位不同讲者各 12 段，正常语速）：

  ```
  同一说话人   中位 0.164   P90 0.279   max 0.386
  不同说话人   中位 0.841   P10 0.763
  可分间隔     +0.484        ← 极干净，阈值 0.40~0.80 全部 24/24 全对
  ```

  阈值 0.90 及以上会把两个真人并成一簇——功能直接失效。

**取 0.6**：正常语音两侧各留 0.2 以上余量，不是走钢丝。代价是明确的——
慢速歌唱类音频会把单人过度切分（实测 1 人拆成 3 簇）。两个域不可调和，
按失败严重性取舍：域内并簇 = 功能失效，域外过切 = 显而易见的错且该域本就不支持。
歌唱类或任何自动定数不理想的场景，用 `n_speakers=` 显式指定人数绕开。

短段（< min_dur）不提取声纹，处理方式见 `assign_speakers` 的 docstring。
"""
import numpy as np

MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"

# 真机双向校准（见模块 docstring）：域内正常语音上，同人距离 P90=0.279、
# 异人 P10=0.763，间隔 0.484；0.40~0.80 任意取值都能 24/24 全对。
# 取 0.6 居中，两侧余量各 0.2 以上。
# tests/fixtures/spk_embeddings_2spk.npy 存的就是那 24 条真实声纹，
# 回归测试直接拿它验证「两个真人必须分得开」——纯 CPU、无需模型。
DEFAULT_THRESHOLD = 0.6


def cluster_speakers(embeddings, n_speakers=None, threshold=DEFAULT_THRESHOLD):
    """纯函数：`[N, 192]`（或任意等长）声纹向量数组 → N 个整数簇标签。

    凝聚层次聚类（average linkage）+ 余弦距离。余弦距离只看向量方向、
    不看幅度——声纹向量的幅度受录音音量、时长这些跟"是谁在说话"无关的
    因素影响，方向才携带说话人信息，这也是说话人确认领域的标准做法。

    `n_speakers` 给定时严格聚成这么多类（若超过样本数则截到样本数，
    不报错、不产生空类）。`n_speakers=None` 时按 `threshold` 自动定簇数：
    average-linkage 的合并距离一旦超过 threshold 就不再合并。

    `threshold` 默认值见模块顶部 `DEFAULT_THRESHOLD` 的注释——按真实单人
    录音校准，不是照抄说话人确认场景的经验阈值。

    边界：N=0 返回空数组；N=1 直接返回 `[0]`（不进聚类器——sklearn 的
    AgglomerativeClustering 至少要 2 个样本）。
    """
    v = np.asarray(embeddings, dtype="float64")
    n = len(v)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    if n_speakers is not None and n_speakers < 1:
        raise ValueError(f"n_speakers 必须 >= 1，收到 {n_speakers}")

    # ⚠️ scikit-learn 目前不在 pyproject.toml 任何依赖列表里（核心/whisper/funasr/dev
    # 都没有），本机能跑是因为环境里恰好装了。`pip install -e ".[dev]"`（CI 用的命令）
    # 装出的干净环境里没有 sklearn，已用隔离 venv 实测复现：12 个测试 ModuleNotFoundError。
    # 加一行 "scikit-learn>=1.3" 到 dependencies 即可（scipy 会作为其依赖一并装上），
    # 不需要改 ci.yml。详见 diarize-report.md「顾虑」一节。
    from sklearn.cluster import AgglomerativeClustering

    if n_speakers is not None:
        k = min(n_speakers, n)
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    else:
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                        metric="cosine", linkage="average")
    return model.fit_predict(v)


def _label_names(labels):
    """整数簇标签 -> "S1"/"S2"/…，按**首次出现的顺序**编号（不是簇 id 本身）。

    调用方必须按时间顺序传入 labels：这样 S1 总是"先开口的那个人"，结果
    人类可读，且不依赖 sklearn 内部簇 id 的编号方式（那是实现细节，不保证
    跨版本/跨数据稳定）。
    """
    names, out = {}, []
    for lab in labels:
        key = int(lab)
        if key not in names:
            names[key] = f"S{len(names) + 1}"
        out.append(names[key])
    return out


def _slice_audio(audio, sr, start, end):
    """按秒切片，越界不报错——截到音频边界；零长度区间返回占位数组而不是崩溃。"""
    a = max(0, int(round(start * sr)))
    b = min(len(audio), int(round(end * sr)))
    if b <= a:
        return np.zeros(1, dtype="float32")
    return audio[a:b]


def _real_embed_fn(clips, sr, device):
    """真实声纹抽取：调 FunASR 本地缓存的 cam++ 模型。真机专用（需要 GPU/CPU
    + funasr 已装 + 模型已缓存），CI 没装 funasr，单测一律走注入的假函数
    （见 tests/test_diarize.py），不会导入到这里。

    真机实测的两个坑（模块 docstring 有背景）：
    1. `spk_embedding` 是 CUDA tensor，必须 `.detach().cpu().numpy()`；
    2. 整批 array 一次性传给 `generate()` 即可，不必逐条调用、不必写临时 wav。
    """
    from funasr import AutoModel

    from .engines import ensure_model
    model = AutoModel(model=ensure_model(MODEL_ID), device=device,
                      disable_update=True)
    r = model.generate(input=list(clips))
    out = []
    for item in r:
        e = item["spk_embedding"]
        e = e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e)
        out.append(e.reshape(-1))
    return np.stack(out)


def assign_speakers(rows, wav, n_speakers=None, device="cuda", min_dur=0.5,
                     threshold=DEFAULT_THRESHOLD, embed_fn=None):
    """给 transcript rows 打 speaker 标签：新增 `"speaker"` 键，就地改 rows 并
    返回同一个列表（`out is rows`）。行数、行序、每条的 `text` 一律不变。

    只加字段，绝不碰 `text`——说话人分离是可选的下游标注，不能成为改动
    正文的理由（项目铁律：新增字段可以，静悄悄改内容不行）。

    三步（模块 docstring 有详细背景）：
      1. 读整段 wav，按每条 row 的 `[start, end]` 切出对应音频；
      2. 对**时长 >= min_dur** 的 row 用 cam++ 抽声纹；
      3. `cluster_speakers()` 聚类，簇按首次出现顺序命名 `S1`/`S2`/…

    **短段（< min_dur，默认 0.5 秒）的处理**：不提取声纹（太短的音频，
    x-vector 类声纹表征本身就不稳定，硬塞进聚类会在向量空间里放一个不像
    任何人的噪声点，尤其容易把自动定人数带偏），改为**继承前一条已知
    标签**；音频开头就是短段、还没有任何已知标签可继承时，向后找第一条
    可靠段兜底；如果整批 row 没有一条达到 min_dur（没有任何锚点可继承），
    全部标 `None`——不编造。

    选"继承"而不是一律标 `None`：说话人轮转发生在几百毫秒的间隙里是小
    概率事件，多数场景下"继承上一条"是合理近似，下游（渲染、字幕）拿到
    连续的标签也比满是 `None` 更可用。代价是明确的、不是没意识到的：
    短插话（"嗯""对"由另一人说出）会被误标成前一说话人的话，这是启发式
    的固有局限，不是 bug。

    `embed_fn`：测试注入点，签名 `embed_fn(clips, sr, device) -> [N, 192]`。
    默认 `None` 时用真实 cam++（`_real_embed_fn`，延迟 import funasr——CI
    没装、也没 GPU，单测一律注入假的，见 tests/test_diarize.py）。
    """
    if not rows:
        return rows

    import soundfile as sf
    audio, sr = sf.read(wav, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)   # 多声道兜底：混成单声道

    order = sorted(range(len(rows)), key=lambda i: (rows[i]["start"], rows[i]["end"]))
    reliable = [i for i in order if rows[i]["end"] - rows[i]["start"] >= min_dur]

    label_map = {}
    if reliable:
        fn = embed_fn or _real_embed_fn
        clips = [_slice_audio(audio, sr, rows[i]["start"], rows[i]["end"]) for i in reliable]
        embeds = np.asarray(fn(clips, sr, device))
        labels = cluster_speakers(embeds, n_speakers=n_speakers, threshold=threshold)
        label_map = dict(zip(reliable, _label_names(labels)))

    # 按时间顺序回填不可靠段：继承前一条已知标签；开头一串短段先攒着，
    # 等遇到第一条可靠段再统一回填；从头到尾都没有可靠段则最后全标 None。
    last, pending = None, []
    for i in order:
        if i in label_map:
            last = label_map[i]
            for p in pending:
                rows[p]["speaker"] = last
            pending = []
            rows[i]["speaker"] = last
        elif last is not None:
            rows[i]["speaker"] = last
        else:
            pending.append(i)
    for p in pending:
        rows[p]["speaker"] = None

    return rows
