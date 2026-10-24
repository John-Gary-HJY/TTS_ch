#!/usr/bin/env bash
# 普通话音素强制对齐（与粤语 mfa.sh 同构）
#
# 前置：
#   1. speech_corpus/<speaker>/ 下每个 <utt>.wav 有同名 .lab（拼音音节，空格分隔）
#   2. 已安装 MFA，且当前目录是 forced_alignment/
#
# 用法：
#   cd forced_alignment
#   bash mfa_mandarin.sh

set -e

DICT="dictionary/pinyin-lexicon-r.txt"
CORPUS="speech_corpus"
OUTPUT="aligned_speech_corpus"

echo "[1/2] 校验语料与发音词典"
mfa validate "$CORPUS" "$DICT"

echo "[2/2] 训练对齐模型并执行对齐"
mfa train \
  --config_path ./mfa.config \
  --clean \
  "$CORPUS" "$DICT" \
  ./mandarin_acoustic_model.zip \
  "$OUTPUT"

echo "完成：对齐结果在 $OUTPUT/"
