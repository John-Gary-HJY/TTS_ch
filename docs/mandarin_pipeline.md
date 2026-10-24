# 普通话 TTS 流水线

本文档描述从"一段录音 + 一份文本稿"到"可合成普通话语音"的完整流程。
粤语基线来自上游 DuriTacoTTS；本分支（TTS_ch）在其基础上增加普通话支持。

## 总览

```
原始录音 + 文本稿
   │
   ├─ ① 文本前端      text/text_frontend.py + text/mandarin.py
   │     汉字 → 拼音 → 音素 → 符号 ID
   │
   ├─ ② 块级粗切      tools/split_blocks.py
   │     按转写稿的 14 个时间戳切成块 → .wav + .lab
   │
   ├─ ③ 句级切分      tools/cut_sentences.py
   │     按句子估算时间切 81 条，切点吸附到能量最低的停顿
   │
   ├─ ④ 强制对齐      MFA（逐句）
   │     句级音素边界 → TextGrid
   │
   ├─ ⑤ 预处理        preprocess.py + preprocess_configs/preprocess_mandarin.yaml
   │     80 维 mel + train_files/train_RenSheng.txt
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
- 数字与符号由 `text/mandarin.py` 规范化

### ② 块级粗切

```bash
.venv/bin/python tools/split_blocks.py --wav ../人声.wav --speaker RenSheng
```

- 单声道化，44.1kHz → 22.05kHz（FFT 低通抗混叠）
- 输出 `forced_alignment/speech_corpus/RenSheng/RS_0001..0014.{wav,lab}`

### ③ 句级切分

```bash
.venv/bin/python tools/cut_sentences.py --speaker RenSheng
```

- 读取 `人声.sentences.tsv` 的句级估算时间
- 在估算点 ±0.5 秒内找能量最低的 20ms 帧作为切点（吸附到停顿）
- 输出 `forced_alignment/sentence_corpus/RenSheng/<utt>_sNN.{wav,lab}`

**为什么不直接对齐块级音频**：14 个块长 23–59 秒，MFA 在 55 秒以上的块上
beam search 会失败（23–32 秒的块可以正常对齐）。逐句后每条 2–19 秒，稳定得多。

### ④ 强制对齐

```bash
export PATH=/124090214/miniconda3/envs/mfa311/bin:$PATH
export MFA_ROOT_DIR=/124090214/cache/mfa_sent
cd forced_alignment

mfa train --single_speaker -j 16 \
  --config_path ./mfa_mandarin_light.config --clean \
  sentence_corpus dictionary/pinyin-lexicon-r.txt \
  ./mandarin_acoustic_model.zip

mfa align --single_speaker -j 16 --clean \
  sentence_corpus dictionary/pinyin-lexicon-r.txt \
  mandarin_acoustic_model.zip aligned_sentences
```

- 用拼音词典自训一个对齐模型，保证音素集与声学模型的 209 个符号一致
- `mfa train` 只产出模型；TextGrid 由随后的 `mfa align` 导出
- 小语料用 `mfa_mandarin_light.config`（10 轮单音素 + 10 轮三音素）

### ⑤ 预处理

```bash
.venv/bin/python preprocess.py --config preprocess_configs/preprocess_mandarin.yaml
```

- 80 维 mel：22.05kHz、`n_fft=1024`、`hop=256`、`fmax=8000`
- 丢弃小于 1 秒或大于 25 秒的样本
- 输出 `preprocessed_data_mandarin/mel/RenSheng/*.npy` 与 `train_files/train_RenSheng.txt`

清单每行 6 个字段，用 `|` 分隔：

```
原始文本|音素起始帧|音素时长|音素序列|mel 路径|说话人
```

**注意**：`train_files/speaker_list.txt` 必须包含说话人名（`RenSheng`），
否则 `data.py` 构建 `speaker_map` 时会 KeyError。

### ⑥ 训练（需要 GPU）

```bash
.venv/bin/python train_duration_predictor.py \
    --data_dir ./train_files --model_dir ./experiments/duration_predictor_zh \
    --training_file train_RenSheng.txt --validation_file val_RenSheng.txt --lan zh

.venv/bin/python train_acoustic.py \
    --data_dir ./train_files --model_dir ./experiments/acoustic_model_zh \
    --training_file train_RenSheng.txt --lan zh
```

### ⑦ 推理

```bash
.venv/bin/python testing.py \
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
| 块级 | 14 块（23–59 秒） |
| 句级 | 81 句（1.9–18.8 秒） |
| MFA 对齐成功 | **78 句**（3 条边界句失败） |
| 预处理通过 | **78 句**（9.5 分钟） |
| 训练 / 验证 | **70 / 8** |
| 说话人 | 1（`RenSheng`） |

**限制**：8.4 分钟只够跑通链路和小样本过拟合，不足以训练稳定音色。
建议扩充到 1–3 小时后再进入训练阶段。

## 相关脚本

| 脚本 | 用途 |
|---|---|
| `tools/audio_qc.py` | 音频体检（采样率、削波、底噪、语音段分布） |
| `tools/split_blocks.py` | 按时间戳粗切，生成块级 MFA 输入 |
| `tools/cut_sentences.py` | 按句级估算时间切分，切点吸附到停顿 |
| `tools/split_sentences.py` | 备用：用块级 TextGrid 细切（块级对齐成功时可用） |
| `tests/test_mandarin.py` | 文本前端回归测试 |

## 环境备注

- MFA 需要 Python ≤3.12；conda 默认装了 3.14 会导致对齐归档写出 0 字节。
  当前使用 `conda create -n mfa311 python=3.11 montreal-forced-aligner`。
- MFA 3.4 的 `mfa train` 不会导出 TextGrid，需要单独跑 `mfa align`。
- 对齐时长与 mel 帧数可能差几帧（每个音素至少算 1 帧的累积舍入），
  `preprocessor.py` 已改为用边缘帧补齐而不是丢弃样本。
