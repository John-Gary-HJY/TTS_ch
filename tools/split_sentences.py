"""把 MFA 对齐后的块级结果细切成句级样本。

输入（MFA 之后）：
    forced_alignment/speech_corpus/<speaker>/<utt>.wav   块级音频（22.05kHz 单声道）
    forced_alignment/aligned_speech_corpus/<speaker>/<utt>.TextGrid
    data_local/transcripts/人声.sentences.tsv            句级文本（含 block_id）

输出（可直接喂给 preprocess.py）：
    forced_alignment/speech_corpus_sentences/<speaker>/<utt>_<nnn>.wav
    forced_alignment/speech_corpus_sentences/<speaker>/<utt>_<nnn>.lab
    forced_alignment/aligned_sentences/<speaker>/<utt>_<nnn>.TextGrid

切分依据：词典把每个音节展开成音素序列，MFA 的 phones 层严格按这个顺序输出，
因此可以逐音素对位，得到每个音节的起止时间；句边界取相邻音节之间静音段的中点。

用法：
    .venv/Scripts/python.exe tools/split_sentences.py --speaker RenSheng --dry-run
    .venv/Scripts/python.exe tools/split_sentences.py --speaker RenSheng
"""

import argparse
import os
import sys
import wave

import numpy as np
import textgrid

SILENCE_MARKS = {"", "sp", "spn", "sil"}
TEXTGRID_TIER = "phones"


def read_wav(path):
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("期望单声道 16-bit wav：{}".format(path))
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, sample_rate


def write_wav(path, data, sample_rate):
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def load_sentences(path):
    """按 block_id 分组返回 {block_id: [(sid, text), ...]}"""
    grouped = {}
    with open(path, encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            sid = int(parts[0])
            block_id = int(parts[1])
            text = parts[-1]
            grouped.setdefault(block_id, []).append((sid, text))
    return grouped


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


def expand_syllables(syllables, lexicon):
    """把音节序列展开成音素序列，返回 (phones, phone_to_syllable_index)。"""
    phones = []
    owner = []
    for index, syllable in enumerate(syllables):
        if syllable not in lexicon:
            raise KeyError("词典缺少音节：{}".format(syllable))
        for phone in lexicon[syllable]:
            phones.append(phone)
            owner.append(index)
    return phones, owner


def align_phones(tier, expected_phones):
    """把 TextGrid 的音素区间对位到期望音素序列。

    返回 (phone_intervals, silences)：
        phone_intervals[i] = (start, end, mark)
        silences = [(start, end), ...]
    """
    phone_intervals = []
    silences = []
    cursor = 0

    for interval in tier:
        mark = (interval.mark or "").strip()
        if mark in SILENCE_MARKS:
            if interval.maxTime > interval.minTime:
                silences.append((interval.minTime, interval.maxTime))
            continue

        if cursor >= len(expected_phones):
            raise ValueError(
                "音素数量超出预期：多余标记 {!r} @ {:.2f}s".format(
                    mark, interval.minTime))

        if mark != expected_phones[cursor]:
            raise ValueError(
                "音素不匹配：第 {} 个期望 {!r}，实际 {!r} @ {:.2f}s".format(
                    cursor, expected_phones[cursor], mark, interval.minTime))

        phone_intervals.append((interval.minTime, interval.maxTime, mark))
        cursor += 1

    if cursor != len(expected_phones):
        raise ValueError(
            "音素数量不足：对齐到 {} 个，期望 {} 个".format(
                cursor, len(expected_phones)))

    return phone_intervals, silences


def write_textgrid(path, intervals, duration):
    grid = textgrid.TextGrid(minTime=0.0, maxTime=duration)
    tier = textgrid.IntervalTier(
        name=TEXTGRID_TIER, minTime=0.0, maxTime=duration)
    for start, end, mark in intervals:
        if end > start:
            tier.add(minTime=start, maxTime=end, mark=mark)
    grid.append(tier)
    grid.write(path)


def main():
    parser = argparse.ArgumentParser(description="按 TextGrid 把块级音频细切成句")
    parser.add_argument("--speaker", default="RenSheng")
    parser.add_argument("--prefix", default="RS")
    parser.add_argument(
        "--corpus-dir",
        default="forced_alignment/speech_corpus")
    parser.add_argument(
        "--textgrid-dir",
        default="forced_alignment/aligned_speech_corpus")
    parser.add_argument(
        "--sentences", default="data_local/transcripts/人声.sentences.tsv")
    parser.add_argument(
        "--overrides",
        default="data_local/transcripts/pronunciation_overrides.tsv")
    parser.add_argument(
        "--out-audio-dir",
        default="forced_alignment/speech_corpus_sentences")
    parser.add_argument(
        "--out-textgrid-dir",
        default="forced_alignment/aligned_sentences")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from text.text_frontend import TextFrontend

    frontend = TextFrontend(lan="zh")
    lexicon = frontend.lexicon
    overrides = load_overrides(args.overrides)
    sentences_by_block = load_sentences(args.sentences)

    audio_root = os.path.join(args.corpus_dir, args.speaker)
    grid_root = os.path.join(args.textgrid_dir, args.speaker)
    out_audio_root = os.path.join(args.out_audio_dir, args.speaker)
    out_grid_root = os.path.join(args.out_textgrid_dir, args.speaker)

    if not args.dry_run:
        os.makedirs(out_audio_root, exist_ok=True)
        os.makedirs(out_grid_root, exist_ok=True)

    total = 0
    for block_id in sorted(sentences_by_block):
        utt_id = "{}_{:04d}".format(args.prefix, block_id)
        wav_path = os.path.join(audio_root, utt_id + ".wav")
        grid_path = os.path.join(grid_root, utt_id + ".TextGrid")

        if not (os.path.exists(wav_path) and os.path.exists(grid_path)):
            print("跳过 {}：缺少 wav 或 TextGrid".format(utt_id))
            continue

        audio, sample_rate = read_wav(wav_path)
        grid = textgrid.TextGrid.fromFile(grid_path)
        tier = grid.getFirst(TEXTGRID_TIER)

        sentence_rows = sentences_by_block[block_id]
        sentence_syllables = []
        for _, text in sentence_rows:
            text = apply_overrides(text, overrides)
            syllables = frontend.text_to_pinyins(text)
            if not syllables:
                raise ValueError("句子无法生成拼音：{}".format(text))
            sentence_syllables.append(syllables)

        all_syllables = [s for group in sentence_syllables for s in group]
        expected_phones, owner = expand_syllables(all_syllables, lexicon)
        phone_intervals, silences = align_phones(tier, expected_phones)

        # 每个音节的时间区间
        syl_start = {}
        syl_end = {}
        for (start, end, _), syllable_index in zip(phone_intervals, owner):
            syl_start.setdefault(syllable_index, start)
            syl_end[syllable_index] = end

        # 每条句子的起止时间
        cursor = 0
        bounds = []
        for group in sentence_syllables:
            first = cursor
            last = cursor + len(group) - 1
            start = syl_start[first]
            end = syl_end[last]
            bounds.append([start, end])
            cursor += len(group)

        # 句间以中点切分，首尾各留 0.05s 余量
        for index in range(len(bounds)):
            if index > 0:
                boundary = 0.5 * (bounds[index - 1][1] + bounds[index][0])
                bounds[index - 1][1] = boundary
                bounds[index][0] = boundary
        bounds[0][0] = max(0.0, bounds[0][0] - 0.05)
        bounds[-1][1] = min(len(audio) / sample_rate, bounds[-1][1] + 0.05)

        for index, ((start, end), syllables) in enumerate(
                zip(bounds, sentence_syllables), start=1):
            utt = "{}_s{:02d}".format(utt_id, index)
            duration = end - start
            if duration < 1.0 or duration > 25.0:
                print("  警告 {}：时长 {:.2f}s 超出 1-25s".format(utt, duration))

            if not args.dry_run:
                start_sample = int(round(start * sample_rate))
                end_sample = int(round(end * sample_rate))
                write_wav(
                    os.path.join(out_audio_root, utt + ".wav"),
                    audio[start_sample:end_sample], sample_rate)
                with open(os.path.join(out_audio_root, utt + ".lab"),
                          "w", encoding="utf-8") as handle:
                    handle.write(" ".join(syllables) + "\n")

                local_intervals = []
                for (p_start, p_end, mark) in phone_intervals:
                    if p_end <= start or p_start >= end:
                        continue
                    local_intervals.append((
                        max(0.0, p_start - start), min(duration, p_end - start), mark))
                write_textgrid(
                    os.path.join(out_grid_root, utt + ".TextGrid"),
                    local_intervals, duration)

            total += 1

        print("{}：{} 句，{} 音节，{} 音素".format(
            utt_id, len(sentence_rows), len(all_syllables), len(expected_phones)))

    print("\n完成：{} 句{}".format(total, "（dry-run，未写文件）" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
