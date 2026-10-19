"""普通话文本规范化。

在把文本交给 pypinyin 之前，先把阿拉伯数字、百分号、日期、英文片段
整理成声学前端能够消费的形式。只依赖标准库，便于单独测试。

设计约束（与仓库现有的 209 音素符号表保持一致）：
- 不改写汉字本身，只处理数字与符号；
- 标点保留原样，由 preprocess_mandarin 统一折叠为一个 sil；
- 英文片段当前词表不支持，移除并打印告警，避免逐字母变成一串 sil。
"""

import re

_DIGITS = "零一二三四五六七八九"
_UNITS = ("", "十", "百", "千")
_BIG_UNITS = ("", "万", "亿", "万亿")

# 会被 preprocess_mandarin 折叠成单个 sil 的标点
PUNCTUATION = set("，。！？；：、,.!?;:…—～~（）()《》〈〉“”\"'‘’[]【】·")

_LATIN_RE = re.compile(r"[A-Za-z]+")
_FULLWIDTH_LATIN_RE = re.compile(r"[Ａ-Ｚａ-ｚ]+")
_FULLWIDTH_DIGIT_RE = re.compile(r"[０-９]")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")
_YEAR_RE = re.compile(r"(\d{4})\s*年")
_DATE_RE = re.compile(r"(\d{1,2})\s*([月日号])")
_DECIMAL_RE = re.compile(r"(\d+\.\d+)")
_INT_RE = re.compile(r"\d+")


def _four_digit_to_chinese(n):
    """把 1..9999 读成中文，例如 1001 -> 一千零一。"""
    out = []
    zero_pending = False
    for pos in (3, 2, 1, 0):
        d = (n // (10 ** pos)) % 10
        if d == 0:
            if out:
                zero_pending = True
            continue
        if zero_pending:
            out.append("零")
            zero_pending = False
        out.append(_DIGITS[d])
        out.append(_UNITS[pos])
    return "".join(out)


def int_to_chinese(n):
    """整数转中文读法（支持到万亿级）。"""
    if n < 0:
        return "负" + int_to_chinese(-n)
    if n == 0:
        return "零"
    if n >= 10 ** 16:
        return digits_to_chinese(str(n))

    groups = []
    while n > 0:
        groups.append(n % 10000)
        n //= 10000

    out = ""
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if g == 0:
            if out and not out.endswith("零"):
                out += "零"
            continue
        if out and g < 1000 and not out.endswith("零"):
            out += "零"
        out += _four_digit_to_chinese(g) + _BIG_UNITS[idx]

    if out.startswith("一十"):
        out = out[1:]
    return out.rstrip("零") or "零"


def digits_to_chinese(s):
    """逐位读，例如 2024 -> 二零二四（用于年份、小数部分）。"""
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for c in s)


def _read_number(s):
    """读取一个数字字符串，支持小数：3.5 -> 三点五。"""
    if "." in s:
        whole, frac = s.split(".", 1)
        return int_to_chinese(int(whole)) + "点" + digits_to_chinese(frac)
    return int_to_chinese(int(s))


def normalize_numbers(text):
    """按中文习惯读数字。

    顺序很重要：百分号 -> 年份 -> 日期 -> 小数 -> 整数。
    年份按逐位读（2024年 -> 二零二四年），其余按数值读。
    """
    text = _PERCENT_RE.sub(lambda m: "百分之" + _read_number(m.group(1)), text)
    text = _YEAR_RE.sub(lambda m: digits_to_chinese(m.group(1)) + "年", text)
    text = _DATE_RE.sub(lambda m: int_to_chinese(int(m.group(1))) + m.group(2), text)
    text = _DECIMAL_RE.sub(lambda m: _read_number(m.group(1)), text)
    text = _INT_RE.sub(lambda m: int_to_chinese(int(m.group(0))), text)
    return text


def normalize_text(text, warn=True):
    """普通话文本规范化入口。

    返回的字符串中不再包含阿拉伯数字和英文字母，但保留标点，
    以便下游统一折叠为静音符号。
    """
    text = (text or "").strip()

    # 全角数字转半角，避免 "２０２４" 绕过数字规则
    text = _FULLWIDTH_DIGIT_RE.sub(
        lambda m: chr(ord(m.group(0)) - 0xFEE0), text)

    text = normalize_numbers(text)

    dropped = _LATIN_RE.findall(text) + _FULLWIDTH_LATIN_RE.findall(text)
    if dropped and warn:
        print("[mandarin] 词表不支持英文，已忽略：{}".format(" ".join(dropped)))
    text = _LATIN_RE.sub(" ", text)
    text = _FULLWIDTH_LATIN_RE.sub(" ", text)

    # 去掉空白（中文不需要词间空格），但保留其他字符
    text = re.sub(r"\s+", "", text)
    return text


def is_punctuation(token):
    """判断 pypinyin 返回的 token 是否为标点。"""
    return bool(token) and all(ch in PUNCTUATION for ch in token)
