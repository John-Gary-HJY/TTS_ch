"""音频体检：给一段录音做进入 TTS 语料前的质量检查。

用法（在仓库根目录）：
    .venv/Scripts/python.exe tools/audio_qc.py ../人声.wav
    .venv/Scripts/python.exe tools/audio_qc.py new_take.wav --json out/qc.json

检查项与判定阈值：
- 采样率：必须是 22050 的整数倍（否则需要重采样）
- 削波：>= 0.1% 的样本触顶视为不合格
- 有效语音：< 60 秒提示语料不足
- 单段时长：统计落在 1-25 秒（项目预处理器的硬限制）的段数
- 底噪：>-50 dBFS 提示环境噪声偏大
"""

import argparse
import json
import os
import sys
import wave

import numpy as np


def read_wav(path):
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        sampwidth = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)

    if sampwidth != 2:
        raise ValueError("只支持 16-bit PCM，当前 sampwidth={}".format(sampwidth))

    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels)
        left, right = data[:, 0], data[:, 1]
        correlation = float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 else 1.0
        data = data.mean(axis=1)
    else:
        correlation = None

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sampwidth": sampwidth,
        "duration": frames / float(sample_rate),
        "data": data,
        "channel_correlation": correlation,
    }


def frame_energy_db(data, sample_rate, frame_sec=0.02):
    frame_len = max(1, int(frame_sec * sample_rate))
    count = len(data) // frame_len
    if count == 0:
        return np.array([-120.0])
    frames = data[: count * frame_len].reshape(count, frame_len)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def segment_speech(db, frame_sec=0.02, threshold=-40.0, min_gap_sec=0.35):
    """按帧能量切出语音段，返回 [(start_sec, end_sec), ...]。"""
    active = np.where(db > threshold)[0]
    if len(active) == 0:
        return []
    max_gap = max(1, int(min_gap_sec / frame_sec))
    segments = []
    start = previous = active[0]
    for index in active[1:]:
        if index - previous > max_gap:
            segments.append((start, previous))
            start = index
        previous = index
    segments.append((start, previous))
    return [((a) * frame_sec, (b + 1) * frame_sec) for a, b in segments]


def band_energy(data, sample_rate, bands):
    n_fft = 2048
    hop = 1024
    if len(data) < n_fft:
        return {}
    windows = np.lib.stride_tricks.sliding_window_view(data, n_fft)[::hop]
    windows = windows * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windows, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    total = spectrum.sum()
    result = {}
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        result["{}-{}Hz".format(low, high)] = round(
            100.0 * spectrum[:, mask].sum() / total, 3) if total else 0.0
    return result


def analyze(path):
    info = read_wav(path)
    data = info["data"]
    sr = info["sample_rate"]

    peak_db = float(20 * np.log10(max(np.abs(data).max(), 1e-9)))
    rms_db = float(20 * np.log10(np.sqrt((data.astype(np.float64) ** 2).mean()) + 1e-9))
    dc_offset = float(data.mean())
    clipped = int((np.abs(data) >= 0.999).sum())

    db = frame_energy_db(data, sr)
    segments = segment_speech(db)
    durations = np.array([end - start for start, end in segments]) if segments else np.array([])

    report = {
        "file": os.path.abspath(path),
        "sample_rate": sr,
        "channels": info["channels"],
        "bit_depth": info["sampwidth"] * 8,
        "duration_sec": round(info["duration"], 2),
        "channel_correlation": None if info["channel_correlation"] is None
        else round(info["channel_correlation"], 4),
        "peak_dbfs": round(peak_db, 2),
        "rms_dbfs": round(rms_db, 2),
        "dc_offset": round(dc_offset, 6),
        "clipped_samples": clipped,
        "clipped_ratio": round(100.0 * clipped / max(len(data), 1), 4),
        "noise_floor_dbfs": round(float(np.percentile(db, 1)), 1),
        "silence_ratio_-40db": round(100.0 * float((db < -40).mean()), 1),
        "speech_segments": int(len(durations)),
        "speech_total_sec": round(float(durations.sum()), 1) if len(durations) else 0.0,
        "segment_median_sec": round(float(np.median(durations)), 2) if len(durations) else 0.0,
        "segments_in_1_25s": int(((durations >= 1) & (durations < 25)).sum()),
        "band_energy_pct": band_energy(data, sr, [
            (0, 300), (300, 1000), (1000, 4000), (4000, 8000), (8000, 11025)]),
    }

    warnings = []
    if sr % 22050 != 0:
        warnings.append("采样率 {} 不是 22050 的整数倍，需要先重采样".format(sr))
    if report["clipped_ratio"] >= 0.1:
        warnings.append("削波比例 {:.3f}% 偏高".format(report["clipped_ratio"]))
    if report["noise_floor_dbfs"] > -50:
        warnings.append("底噪 {:.1f} dBFS 偏高".format(report["noise_floor_dbfs"]))
    if report["speech_total_sec"] < 60:
        warnings.append("有效语音仅 {:.1f}s，用于训练偏少".format(report["speech_total_sec"]))
    if report["segments_in_1_25s"] == 0:
        warnings.append("没有落在 1-25 秒的语音段")
    report["warnings"] = warnings
    return report


def main():
    parser = argparse.ArgumentParser(description="TTS 语料音频体检")
    parser.add_argument("audio", help="待检查的 wav 路径")
    parser.add_argument("--json", help="把结果写入 JSON 文件")
    args = parser.parse_args()

    report = analyze(args.audio)

    print("文件        : {}".format(report["file"]))
    print("格式        : {} Hz / {} bit / {} 声道".format(
        report["sample_rate"], report["bit_depth"], report["channels"]))
    print("总时长      : {:.2f} s".format(report["duration_sec"]))
    print("有效语音    : {:.1f} s（{} 段，中位 {:.2f} s，1-25s 内 {} 段）".format(
        report["speech_total_sec"], report["speech_segments"],
        report["segment_median_sec"], report["segments_in_1_25s"]))
    print("电平        : 峰值 {:.2f} dBFS，RMS {:.2f} dBFS".format(
        report["peak_dbfs"], report["rms_dbfs"]))
    print("质量        : 削波 {:.4f}%，直流 {:.6f}，底噪 {:.1f} dBFS".format(
        report["clipped_ratio"], report["dc_offset"], report["noise_floor_dbfs"]))
    if report["channel_correlation"] is not None:
        print("声道相关    : {:.4f}".format(report["channel_correlation"]))
    print("频带能量    : {}".format(
        "  ".join("{}={}%".format(k, v) for k, v in report["band_energy_pct"].items())))

    if report["warnings"]:
        print("\n提示：")
        for warning in report["warnings"]:
            print("  - {}".format(warning))
    else:
        print("\n结论：未发现阻塞性问题")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("\n已写入 {}".format(args.json))

    return 0


if __name__ == "__main__":
    sys.exit(main())
