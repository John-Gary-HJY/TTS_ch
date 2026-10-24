"""按时间戳把整段录音切成块，并生成 MFA 需要的 .wav + .lab。

这一步只做"粗切"：按转写稿里的 14 个时间戳块切分。块内还会包含多句，
等 MFA 对齐完成后再按 TextGrid 的句边界细切（见 tools/split_sentences.py）。

用法（在仓库根目录）：
    .venv/Scripts/python.exe tools/split_blocks.py \
        --wav ../人声.wav \
        --blocks data_local/transcripts/人声.blocks.txt \
        --speaker RenSheng

产物：
    forced_alignment/speech_corpus/<speaker>/<utt_id>.wav  (单声道 22.05kHz)
    forced_alignment/speech_corpus/<speaker>/<utt_id>.lab  (拼音音节，空格分隔)
    data_local/transcripts/blocks_manifest.tsv
"""

import argparse
import os
import re
import sys
import wave

import numpy as np

BLOCK_RE = re.compile(r"^\[(\d+):(\d+)-(\d+):(\d+)\]\s*(.+)$")


def read_wav(path):
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        sampwidth = handle.getsampwidth()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    if sampwidth != 2:
        raise ValueError("只支持 16-bit PCM，当前 sampwidth={}".format(sampwidth))

    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def lowpass_fft(data, sample_rate, pass_hz, stop_hz):
    """零相位低通，用于降采样前抗混叠。"""
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
    gain = np.ones_like(freqs)
    gain[freqs >= stop_hz] = 0.0
    transition = (freqs > pass_hz) & (freqs < stop_hz)
    phase = (freqs[transition] - pass_hz) / (stop_hz - pass_hz)
    gain[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return np.fft.irfft(spectrum * gain, n=len(data))


def resample(data, sample_rate, target_rate):
    if sample_rate == target_rate:
        return data
    if sample_rate % target_rate != 0:
        raise ValueError(
            "只支持整数倍降采样：{} -> {}".format(sample_rate, target_rate))
    factor = sample_rate // target_rate
    filtered = lowpass_fft(
        data, sample_rate, pass_hz=0.90 * target_rate / 2.0,
        stop_hz=target_rate / 2.0)
    return filtered[::factor]


def write_wav(path, data, sample_rate):
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def parse_blocks(path):
    blocks = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = BLOCK_RE.match(line)
            if not match:
                continue
            start = int(match.group(1)) * 60 + int(match.group(2))
            end = int(match.group(3)) * 60 + int(match.group(4))
            blocks.append({
                "start": start,
                "end": end,
                "text": match.group(5).strip(),
            })
    return blocks


def load_overrides(path):
    overrides = []
    if not path or not os.path.exists(path):
        return overrides
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                overrides.append((parts[0], parts[1]))
    return overrides


def apply_overrides(text, overrides):
    for source, target in overrides:
        text = text.replace(source, target)
    return text


def main():
    parser = argparse.ArgumentParser(description="按时间戳切分录音并生成 MFA 输入")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--blocks", default="data_local/transcripts/人声.blocks.txt")
    parser.add_argument("--out-dir", default="forced_alignment/speech_corpus")
    parser.add_argument("--speaker", default="RenSheng")
    parser.add_argument("--prefix", default="RS")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument(
        "--overrides", default="data_local/transcripts/pronunciation_overrides.tsv")
    parser.add_argument(
        "--manifest", default="data_local/transcripts/blocks_manifest.tsv")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from text.text_frontend import TextFrontend

    frontend = TextFrontend(lan="zh")
    overrides = load_overrides(args.overrides)
    if overrides:
        print("已加载 {} 条读音覆盖：{}".format(
            len(overrides), ", ".join("{}→{}".format(a, b) for a, b in overrides)))

    audio, source_rate = read_wav(args.wav)
    print("输入：{} Hz，{:.2f} s，{} 个块".format(
        source_rate, len(audio) / source_rate, len(parse_blocks(args.blocks))))

    audio = resample(audio, source_rate, args.sr)
    print("重采样：{} Hz -> {} Hz".format(source_rate, args.sr))

    speaker_dir = os.path.join(args.out_dir, args.speaker)
    os.makedirs(speaker_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)

    rows = []
    for index, block in enumerate(parse_blocks(args.blocks), start=1):
        utt_id = "{}_{:04d}".format(args.prefix, index)
        start = int(round(block["start"] * args.sr))
        end = int(round(min(block["end"], len(audio) / args.sr) * args.sr))
        clip = audio[start:end]

        text = apply_overrides(block["text"], overrides)
        pinyins = frontend.text_to_pinyins(text)
        if not pinyins:
            print("跳过 {}：无法生成拼音".format(utt_id))
            continue

        wav_path = os.path.join(speaker_dir, utt_id + ".wav")
        lab_path = os.path.join(speaker_dir, utt_id + ".lab")
        write_wav(wav_path, clip, args.sr)
        with open(lab_path, "w", encoding="utf-8") as handle:
            handle.write(" ".join(pinyins) + "\n")

        rows.append((
            utt_id, block["start"], block["end"],
            round(len(clip) / args.sr, 2), len(block["text"]),
            len(pinyins), " ".join(pinyins)))

        print("{}  {:>3.1f}s  {:>3d}字  {:>3d}音节".format(
            utt_id, len(clip) / args.sr, len(block["text"]), len(pinyins)))

    with open(args.manifest, "w", encoding="utf-8") as handle:
        handle.write("utt_id\tstart_sec\tend_sec\tduration_sec\tchars\tpinyin_count\tpinyin\n")
        for row in rows:
            handle.write("\t".join(str(item) for item in row) + "\n")

    print("\n完成：{} 个块 -> {}".format(len(rows), speaker_dir))
    print("清单：{}".format(args.manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
