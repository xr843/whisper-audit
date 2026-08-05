"""ASR 后端抽象。

引入第二个引擎的目的是跨模型互补——实测同模型不同 chunk 的双路净贡献只有
1,792 字（全文 4%），却要付 +31 分钟。跨引擎的互补性应当远大于此，但这是
待验证的假设，由 CER 裁决。
"""


class Engine:
    name = "base"

    def transcribe(self, wav, **opts):
        """返回 {"duration": float, "segments": [...]}。

        segment 必须有 start/end/text；avg_logprob、no_speech_prob、words 可缺。
        words 缺失时下游自动降级：字幕退回整段、段内标点不加。
        """
        raise NotImplementedError


_REGISTRY = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_engine(name, **kw):
    if name not in _REGISTRY:
        raise KeyError(f"未知引擎 {name}，已注册：{sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)
