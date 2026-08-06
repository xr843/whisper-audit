#!/usr/bin/env bash
# 获取 AISHELL-1 的 test 划分，生成 bench/aishell_test.jsonl 作为公开基准。
#
#   bash bench/fetch_aishell_test.sh [目标目录]
#
# ⚠️ AISHELL-1 只提供 15GB 全量包（train+dev+test 在一个 tgz 里），没有单独的
# test 下载。本机走代理，很可能下不动或极慢。**下不动就不要卡在这里**——
# 公开集是「可复现的对外数字」，不是项目能否工作的前提。真实 CER 以自建金标为准。
#
# 下不动时的替代：evaluate 是 manifest 驱动的，任何「音频 + 正确文本」的配对
# 都能评。自己攒一份 jsonl 即可：
#   {"audio": "/path/a.wav", "text": "这一条的正确文本"}
set -euo pipefail

DIR="${1:-bench/data/aishell}"
OUT="bench/aishell_test.jsonl"
URL="https://openslr.magicdatatech.com/resources/33/data_aishell.tgz"

if [ -f "$OUT" ]; then
  echo "已存在 $OUT（$(wc -l < "$OUT") 条），跳过。要重建请先删除它。"
  exit 0
fi

mkdir -p "$DIR"
TGZ="$DIR/data_aishell.tgz"

echo "下载 $URL"
echo "约 15GB，支持断点续传；中断后重跑本脚本即可继续。"
if ! curl -fL --retry 3 --retry-delay 5 -C - -o "$TGZ.part" "$URL"; then
  echo "❌ 下载失败。本机走代理时这很常见。" >&2
  echo "   不要在此阻塞：跳过公开集，用自建金标（audio-transcribe goldset）继续。" >&2
  echo "   已把半成品留在 $TGZ.part，可重跑续传。" >&2
  exit 1
fi
mv "$TGZ.part" "$TGZ"

echo "解包（只取 test 与 transcript）…"
tar -xzf "$TGZ" -C "$DIR" \
    --wildcards 'data_aishell/transcript/*' 'data_aishell/wav/test.tar.gz' 2>/dev/null \
  || tar -xzf "$TGZ" -C "$DIR"
find "$DIR" -name 'test.tar.gz' -exec tar -xzf {} -C "$DIR" \;

TRANS=$(find "$DIR" -name 'aishell_transcript_*.txt' | head -1)
[ -n "$TRANS" ] || { echo "❌ 找不到 transcript 文件" >&2; exit 1; }

python3 - "$DIR" "$TRANS" "$OUT" <<'PY'
import json, os, sys
root, trans, out = sys.argv[1:4]

# transcript 每行： BAC009S0764W0121 甚 至 出 现 交 易 几 乎 停 滞 的 情 况
text = {}
for line in open(trans, encoding="utf-8"):
    parts = line.split()
    if len(parts) >= 2:
        text[parts[0]] = "".join(parts[1:])

wavs = {}
for dirpath, _, names in os.walk(root):
    if os.sep + "test" + os.sep not in dirpath + os.sep:
        continue
    for n in names:
        if n.endswith(".wav"):
            wavs[n[:-4]] = os.path.join(dirpath, n)

n = 0
with open(out, "w", encoding="utf-8") as f:
    for k, path in sorted(wavs.items()):
        if k in text:
            f.write(json.dumps({"audio": path, "text": text[k]},
                               ensure_ascii=False) + "\n")
            n += 1
print(f"生成 {out}：{n} 条")
if n == 0:
    raise SystemExit("❌ 一条都没匹配上，检查解包结构")
PY
