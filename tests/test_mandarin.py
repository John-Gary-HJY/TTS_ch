"""普通话文本前端回归测试。

运行方式（在仓库根目录）：
    .venv/Scripts/python.exe -m unittest discover -s tests -v
"""

import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from text.mandarin import int_to_chinese, normalize_text
from text.text_frontend import TextFrontend

TRANSCRIPT = os.path.join(ROOT, "data_local", "transcripts", "人声.sentences.tsv")


class TestIntToChinese(unittest.TestCase):
    def test_common_numbers(self):
        cases = {
            0: "零",
            1: "一",
            10: "十",
            11: "十一",
            20: "二十",
            21: "二十一",
            101: "一百零一",
            110: "一百一十",
            365: "三百六十五",
            1001: "一千零一",
            1010: "一千零一十",
            2024: "二千零二十四",
            10010: "一万零一十",
            100000: "十万",
            1000000: "一百万",
            100000001: "一亿零一",
        }
        for number, expected in cases.items():
            self.assertEqual(int_to_chinese(number), expected, msg=str(number))

    def test_negative(self):
        self.assertEqual(int_to_chinese(-5), "负五")


class TestNormalizeText(unittest.TestCase):
    def test_year_is_read_digit_by_digit(self):
        self.assertEqual(
            normalize_text("2024年有365天。"),
            "二零二四年有三百六十五天。")

    def test_decimal(self):
        self.assertEqual(normalize_text("费用是3.5元。"), "费用是三点五元。")

    def test_percent(self):
        self.assertEqual(normalize_text("增长了50%。"), "增长了百分之五十。")

    def test_date(self):
        self.assertEqual(normalize_text("12月31日"), "十二月三十一日")

    def test_fullwidth_digits(self):
        self.assertEqual(normalize_text("２０２４年", warn=False), "二零二四年")

    def test_english_is_dropped(self):
        self.assertEqual(normalize_text("你好，world！", warn=False), "你好，！")


class TestTextFrontendZh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf = TextFrontend(lan="zh")

    def test_symbol_inventory(self):
        self.assertEqual(len(self.tf.SYMBOLS), 211)
        self.assertEqual(self.tf.SYMBOLS[0], "_")
        self.assertEqual(self.tf.SYMBOLS[1], "sil")

    def test_roundtrip_is_non_empty(self):
        out = self.tf.backward(self.tf.text_to_phonemes("今天天气很好。"))
        self.assertTrue(out.startswith("sil "))
        self.assertTrue(out.endswith(" sil"))
        self.assertIn("in1", out)

    def test_consecutive_punctuation_collapses(self):
        out = self.tf.backward(self.tf.text_to_phonemes("你好，，世界！"))
        self.assertEqual(out.count("sil"), 3)

    def test_ids_are_in_range(self):
        for text in ("这是一个测试", "你好。", "2024年。"):
            ids = self.tf.text_to_phonemes(text)
            self.assertGreater(len(ids), 2, msg=text)
            self.assertTrue(all(0 <= i < len(self.tf.SYMBOLS) for i in ids), msg=text)

    def test_lexicon_phones_are_all_known_symbols(self):
        symbols = set(self.tf.SYMBOLS)
        for key, phones in self.tf.lexicon.items():
            for phone in phones:
                self.assertIn(
                    phone, symbols,
                    msg="词典条目 {} 的音素 {} 不在符号表中".format(key, phone))


class TestCorpusSentences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf = TextFrontend(lan="zh")

    @unittest.skipUnless(os.path.exists(TRANSCRIPT), "本地语料转写不存在，跳过")
    def test_every_sentence_converts(self):
        with open(TRANSCRIPT, encoding="utf-8") as handle:
            next(handle)
            rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]

        self.assertEqual(len(rows), 81)
        for row in rows:
            text = row[-1]
            with contextlib.redirect_stdout(io.StringIO()):
                ids = self.tf.text_to_phonemes(text)
            self.assertGreater(len(ids), 0, msg=text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
