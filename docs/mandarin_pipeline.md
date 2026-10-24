# 普通话 TTS 流水线

本文档描述这个仓库从"一段录音 + 一份文本稿"到"可合成普通话语音"的完整流程。
粤语基线来自上游 DuriTacoTTS；本分支（TTS_ch）在其基础上增加普通话支持。

## 总览

```
原始录音 + 文本稿
   │
   ├─ ① 文本前端      text/text_frontend.py + text/mandarin.py
   │     汉字 → 拼音 → 音素 → 符号 ID
   │
   ├─ ② 粗切          tools/split_blocks.py
   │     按时间戳把整段录音切成块，生成 .wav + .lab（拼音）
   │
   ├─ ③ 强制对齐      MFA + forced_alignment/mfa_mandarin.sh
   │     音素 → 每帧边界（TextGrid）
   │
   ├─ ④ 细切          tools/split_sentences.py
   │     按 TextGrid 的句边界切成 1–25 秒的句级样本
   │
   ├─ ⑤ 预处理        preprocess.py + preprocess_configs/preprocess_mandarin.yaml
   │     算 80 维 mel，生成 train_files/train_*.txt
   │
   ├─ ⑥ 训练          时长预测器 + 声学模型（需要 GPU）
   │
   └─ ⑦ 推理          testing.py → HiFi-GAN → wav
```

## 各步骤说明

### ① 文本前端

入口：`TextFrontend(lan="zh")`

- 符号表：`[PAD, SIL] + 209 个拼音符号 = 211`
  - PAD（`_`，ID 0）：批处理填充，训练时被 mask
  - SIL（`sil`，ID 1）：真实静音，有音频、有时长
- 拼音符号：23 个声母 + 185 个带调韵母 + 1 个儿化后缀 `rr`
- 词典：`forced_alignment/dictionary/pinyin-lexicon-r.txt`（4115 个音节键）

数字与符号由 `text/mandarin.py` 规范化，例如：

| 原文 | 规范化 |
|---|---|
| `2024年` | `二零二四年` |
| `365天` | `三百六十五天` |
| `3.5元` | `三点五元` |
| `50%` | `百分之五十` |

英文片段当前词表不支持，会被移除并打印告警。

### ② 粗切

```bash
.venv/Scripts/python.exe tools/split_blocks.py \
    --wav ../人声.wav \
    --speaker RenSheng
```

- 输入：整段 wav + `data_local/transcripts/人声.blocks.txt`
- 处理：降混为单声道，44.1kHz → 22.05kHz（FFT 低通抗混叠）
- 输出：`forced_alignment/speech_corpus/RenSheng/RS_0001.wav` + `.lab`
- `.lab` 内容是拼音音节，空格分隔（MFA 的输入）

### ③ 强制对齐

```bash
cd forced_alignment
mfa validate speech_corpus dictionary/pinyin-lexicon-r.txt
mfa train --config_path ./mfa.config --clean \
    speech_corpus dictionary/pinyin-lexicon-r.txt \
    ./mandarin_acoustic_model.zip aligned_speech_corpus
```

用拼音词典自训一个对齐模型，保证对齐输出的音素集与声学模型的 209 个符号完全一致。
输出 `aligned_speech_corpus/<speaker>/<utt>.TextGrid`。

### ④ 细切

```bash
.venv/Scripts/python.exe tools/split_sentences.py --speaker RenSheng --dry-run
.venv/Scripts/python.exe tools/split_sentences.py --speaker RenSheng
```

- 词典把每个音节展开成音素序列，MFA 的 `phones` 层严格按这个顺序输出
- 逐音素对位后得到每个音节的起止时间
- 句边界取相邻音节之间静音段的中点
- 输出：
  - `forced_alignment/speech_corpus_sentences/<speaker>/<utt>_sNN.wav` + `.lab`
  - `forced_alignment/aligned_sentences/<speaker>/<utt>_sNN.TextGrid`

### ⑤ 预处理

```bash
.venv/Scripts/python.exe preprocess.py --config preprocess_configs/preprocess_mandarin.yaml
```

- 生成 80 维 mel：22.05kHz、`n_fft=1024`、`hop=256`、`fmax=8000`
- 丢弃小于 1 秒或大于 25 秒的样本
- 输出 `preprocessed_data_mandarin/mel/<speaker>/*.npy` 与 `train_files/train_RenSheng.txt`

训练清单每行 6 个字段，用 `|` 分隔：

```
原始文本|音素起始帧|音素时长|音素序列|mel 路径|说话人
```

### ⑥ 训练（需要 GPU）

```bash
.venv/Scripts/python.exe train_duration_predictor.py \
    --data_dir ./train_files --model_dir ./experiments/duration_predictor_zh \
    --training_file train_RenSheng.txt --validation_file val_RenSheng.txt --lan zh

.venv/Scripts/python.exe train_acoustic.py \
    --data_dir ./train_files --model_dir ./experiments/acoustic_model_zh \
    --training_file train_RenSheng.txt --lan zh
```

### ⑦ 推理

```bash
.venv/Scripts/python.exe testing.py \
    --model_dir ./experiments/acoustic_model_zh \
    --dp_model_dir ./experiments/duration_predictor_zh \
    --restore_file <acoustic checkpoint> \
    --dp_restore_file <duration checkpoint> \
    --vocoder_dir ./experiments/pretrained_hifigan_vocoder/UNIVERSAL_V1 \
    --fpath test_script/test_zh.txt --speaker RenSheng --lan zh
```

## 当前语料

| 项目 | 数值 |
|---|---|
| 原始录音 | `人声.wav`，44.1kHz / 16-bit / 双声道，588.47 秒 |
| 有效语音 | 504.8 秒（8 分 25 秒） |
| 切块 | 14 块（23–59 秒） |
| 分句 | 81 句（估算 1.9–18.8 秒） |
| 音节总数 | 2813 |
| 说话人 | 1（`RenSheng`） |

**限制**：8.4 分钟只够跑通链路和小样本过拟合，不足以训练稳定音色。
建议扩充到 1–3 小时后再进入训练阶段。

## 相关脚本

| 脚本 | 用途 |
|---|---|
| `tools/audio_qc.py` | 音频体检（采样率、削波、底噪、语音段分布） |
| `tools/split_blocks.py` | 按时间戳粗切，生成 MFA 输入 |
| `tools/split_sentences.py` | 按 TextGrid 细切成句 |
| `tests/test_mandarin.py` | 文本前端回归测试 |
