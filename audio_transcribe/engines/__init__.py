"""ASR 后端抽象。

引入第二个引擎的初衷是跨模型互补。**2026-08-06 在慢速歌唱域实测的裁决是：不并**——
whisper 双路 74.2% CER，掺入 funasr 后 84~86.5%，全面更差。机理：两个引擎的
失败模式相反（whisper 不敢出字 dele=2190，funasr 什么都敢出但大量错 sub=2352），
而 combine 的逐桶字数竞赛假设「字多=转得全」，funasr 的高错误文本每桶都字多、
整桶通吃，把 whisper 较准的内容挤掉。讲座域（正常语音）待测，需要讲座域金标。
详见 docs/measurements.md。
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
        # 注册靠模块 import 时的 @register 副作用；引擎模块是按需加载的
        # （funasr 依赖不装也不能拖垮包），所以这里先尝试延迟导入。
        import importlib
        try:
            importlib.import_module(f".{name}", __package__)
        except ImportError:
            pass
    if name not in _REGISTRY:
        raise KeyError(f"未知引擎 {name}，已注册：{sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)
