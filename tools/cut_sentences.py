"""按句级估算时间切分块级音频，供 MFA 逐句对齐。

为什么不直接对块级音频做对齐：
    14 个块时长 23-59 秒，MFA 在 55 秒以上的块上 beam search 会失败
    （23-32 秒的块可以正常对齐）。逐句切分后每条只有 2-19 秒，稳定得多。

边界怎么定：
    sentences.tsv 的 est_start/est_end 是按字符数比例估算的，会有偏差。
    因此在估算点 ±0.5 秒内找能量最低的 20ms 帧作为切点——把刀口"吸附"
    到停顿上，避免切在字中间。

用法：
    .venv/Scripts/python.exe tools/cut_sentences.py --speaker RenSheng --dry-run
    .venv/Scripts/python.exe tools/cut_sentences.py --speaker RenSheng

输出：
    forced_alignment/sentence_corpus/<speaker>/<utt>_sNN.wav
    forced_alignment/sentence_corpus/<speaker>/<utt>_sNN.lab
    data_local/transcripts/sentence_cuts.tsv
"""

import argparse
import os
import sys
import wave

import numpy as np

FRAME_SEC = 0.02


def read_wav(path):
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("期望单声道 16-bit wav：{}".format(path))
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return data, sample_rate


def write_wav(path, data, sample_rate):
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def load_sentences(path):
    grouped = {}
    with open(path, encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            grouped.setdefault(int(parts[1]), []).append({
                "sent_id": int(parts[0]),
                "est_start": float(parts[4]),
                "est_end": float(parts[5]),
                "text": parts[-1],
            })
    return grouped


def load_block_starts(path):
    starts = {}
    with open(path, encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            starts[parts[0]] = float(parts[1])
    return starts


def load_overrides(path):
    pairs = []
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    return pairs


def apply_overrides(text, pairs):
    for source, target in pairs:
        text = text.replace(source, target)
    return text


def frame_energy(data, sample_rate):
    frame_len = max(1, int(FRAME_SEC * sample_rate))
    count = len(data) // frame_len
    frames = data[: count * frame_len].reshape(count, frame_len)
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)


def snap_to_pause(energy, target, window_sec):
    """在 target ± window 内找能量最低的帧，返回该帧中心时间。"""
    lo = max(0, int((target - window_sec) / FRAME_SEC))
    hi = min(len(energy), int((target + window_sec) / FRAME_SEC) + 1)
    if hi <= lo:
        return target
    index = lo + int(np.argmin(energy[lo:hi]))
    return (index + 0.5) * FRAME_SEC


def main():
    parser = argparse.ArgumentParser(description="按句级估算时间切分块级音频")
    parser.add_argument("--speaker", default="RenSheng")
    parser.add_argument("--prefix", default="RS")
    parser.add_argument(
        "--corpus-dir", default="forced_alignment/speech_corpus")
    parser.add_argument(
        "--sentences", default="data_local/transcripts/人声.sentences.tsv")
    parser.add_argument(
        "--blocks-manifest",
        default="data_local/transcripts/blocks_manifest.tsv")
    parser.add_argument(
        "--overrides",
        default="data_local/transcripts/pronunciation_overrides.tsv")
    parser.add_argument(
        "--out-dir", default="forced_alignment/sentence_corpus")
    parser.add_argument(
        "--manifest", default="data_local/transcripts/sentence_cuts.tsv")
    parser.add_argument("--window", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from text.text_frontend import TextFrontend

    frontend = TextFrontend(lan="zh")
    overrides = load_overrides(args.overrides)
    sentences = load_sentences(args.sentences)
    block_starts = load_block_starts(args.blocks_manifest)

    audio_root = os.path.join(args.corpus_dir, args.speaker)
    out_root = os.path.join(args.out_dir, args.speaker)
    if not args.dry_run:
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest)),
                    exist_ok=True)

    rows = []
    for block_id in sorted(sentences):
        utt_id = "{}_{:04d}".format(args.prefix, block_id)
        wav_path = os.path.join(audio_root, utt_id + ".wav")
        if not os.path.exists(wav_path):
            print("跳过 {}：找不到 {}".format(utt_id, wav_path))
            continue

        audio, sample_rate = read_wav(wav_path)
        duration = len(audio) / sample_rate
        energy = frame_energy(audio, sample_rate)
        block_start = block_starts[utt_id]

        items = sentences[block_id]
        starts = [max(0.0, item["est_start"] - block_start) for item in items]
        ends = [min(duration, item["est_end"] - block_start) for item in items]

        # 相邻句子的公共边界吸附到停顿
        boundaries = []
        for index in range(len(items) - 1):
            raw = 0.5 * (ends[index] + starts[index + 1])
            snapped = snap_to_pause(energy, raw, args.window)
            lower = starts[index] + 0.2
            upper = ends[index + 1] - 0.2
            boundaries.append(min(max(snapped, lower), upper))

        cuts = []
        for index, item in enumerate(items):
            start = starts[index] if index == 0 else boundaries[index - 1]
            end = ends[index] if index == len(items) - 1 else boundaries[index]
            cuts.append((start, end, item))

        for index, (start, end, item) in enumerate(cuts, start=1):
            utt = "{}_s{:02d}".format(utt_id, index)
            pinyins = frontend.text_to_pinyins(
                apply_overrides(item["text"], overrides))
            if not pinyins:
                print("  跳过 {}：无法生成拼音".format(utt))
                continue
            clip_duration = end - start
            if clip_duration < 1.0 or clip_duration > 25.0:
                print("  警告 {}：{:.2f}s 超出 1-25s".format(utt, clip_duration))

            if not args.dry_run:
                write_wav(
                    os.path.join(out_root, utt + ".wav"),
                    audio[int(round(start * sample_rate)):
                          int(round(end * sample_rate))],
                    sample_rate)
                with open(os.path.join(out_root, utt + ".lab"),
                          "w", encoding="utf-8") as handle:
                    handle.write(" ".join(pinyins) + "\n")

            rows.append((utt, block_id, item["sent_id"],
                         round(start, 2), round(end, 2),
                         round(clip_duration, 2), len(item["text"]),
                         len(pinyins), item["text"]))

        print("{}：{} 句".format(utt_id, len(cuts)))

    if not args.dry_run:
        with open(args.manifest, "w", encoding="utf-8") as handle:
            handle.write(
                "utt_id\tblock_id\tsent_id\tstart_sec\tend_sec\t"
                "duration_sec\tchars\tpinyin_count\ttext\n")
            for row in rows:
                handle.write("\t".join(str(x) for x in row) + "\n")

    total_duration = sum(row[5] for row in rows)
    print("\n完成：{} 句，共 {:.1f} 秒{}".format(
        len(rows), total_duration,
        "（dry-run，未写文件）" if args.dry_run else ""))
    print("清单：{}".format(args.manifest))


if __name__ == "__main__":
    sys.exit(main())
