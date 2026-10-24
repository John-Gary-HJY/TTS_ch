from g2p_en import G2p
from pypinyin import pinyin, Style

from string import punctuation
import re

from .cleaners import english_cleaners
from .mandarin import normalize_text
from .symbols import _arpabet, _pinyin, _jyutping

import ToJyutping


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, lexicon):
    text = english_cleaners(text)
    text = text.rstrip(punctuation)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    # mark phone boundary with {}
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sil}", phones)
    phones = phones.replace("}{", " ")[1:-1].split()

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))

    return phones


def preprocess_mandarin(text, lexicon, silence=True):
    raw_text = text
    text = normalize_text(text)
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    phones = []
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        elif silence:
            # 标点、空白和词典外的字符统一折叠成一个 sil，避免连续静音
            if not phones or phones[-1] != "sil":
                phones.append("sil")

    if silence:
        print("Raw Text Sequence: {}".format(raw_text))
        print("Normalized Text Sequence: {}".format(text))
        print("Pinyin Sequence: {}".format(pinyins))
        print("Phoneme Sequence: {}".format(phones))

    return phones


def preprocess_canton(text, lexicon, silence=True):
    jyutpings = ToJyutping.get_jyutping_text(text)
    tokens = jyutpings.split()
    if not silence:
        jyutpings = [re.sub(r'[^a-z0-9]', '', t) for t in tokens]
        jyutpings = [t for t in jyutpings if t]
    else:
        jyutpings = []
        for t in tokens:
            clean = re.sub(r'[^a-z0-9]', '', t)
            if clean:
                jyutpings.append(clean)
            if re.search(r'[,，。.!！?？；;]', t):
                jyutpings.append('sil')

    phones = []
    for p in jyutpings:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sil")
    if silence:
        print("Raw Text Sequence: {}".format(text))
        print("jyutping Sequence: {}".format(jyutpings))
        print("Phoneme Sequence: {}".format(phones))

    return jyutpings, phones


class TextFrontend(object):
    """
    Provides interface for conversion of symbolic phoneme text representation
    into a sequence of embedding ids for pushing them as inputs to TTS frontend.
    """

    def __init__(self, lan='en'):
        # PAD should be first to have zero id
        self.PAD = '_'
        self.SIL = 'sil'
        self.lan = lan
        self.translator = None
        if self.lan == 'en':
            self.PHONEME_CODES = _arpabet
            self.lexicon = read_lexicon("./forced_alignment/dictionary/librispeech-lexicon.txt")
            self.SYMBOLS = [self.PAD] + [self.SIL] + self.PHONEME_CODES
        elif self.lan == 'yue':
            self.PHONEME_CODES = _jyutping
            self.lexicon = read_lexicon("./forced_alignment/dictionary/canton-lexicon.txt")
            # in order to be consistent with previous trained models, sil is not the second symbol for cantonese
            self.SYMBOLS = [self.PAD] + self.PHONEME_CODES
        else:
            self.PHONEME_CODES = _pinyin
            self.lexicon = read_lexicon("./forced_alignment/dictionary/pinyin-lexicon-r.txt")
            self.SYMBOLS = [self.PAD] + [self.SIL] + self.PHONEME_CODES

        self._PHONEME_SEP = ' '
        self._symbol_to_id = {s: i for i, s in enumerate(self.SYMBOLS)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.SYMBOLS)}

    def _should_keep_token(self, token, token_dict):
        return token in token_dict \
            and token != self.PAD and token != self._symbol_to_id[self.PAD]

    def forward(self, string):
        string = string.split(self._PHONEME_SEP) if isinstance(
            string, str) else string
        
        sequence = [self._symbol_to_id[s] for s in string
                    if self._should_keep_token(s, self._symbol_to_id)]
        return sequence

    def backward(self, sequence: list):
        string = [self._id_to_symbol[idx] for idx in sequence
                  if self._should_keep_token(idx, self._id_to_symbol)]
        string = self._PHONEME_SEP.join(string)
        return string

    def text_to_phonemes(self, text):
        """
        Convert text into ARPAbet.
        For known words use CMUDict; for the rest try 'espeak' (to IPA) followed by 'listener'.
        :param text: str, input text.
        :param lan: str, language code
        :return: list of str, phonemes
        """
        if self.lan == "en":
            phones = preprocess_english(text, self.lexicon)
        elif self.lan == "yue":
            _, phones = preprocess_canton(text, self.lexicon)
        else:
            phones = preprocess_mandarin(text, self.lexicon)

        if phones[0] != 'sil':
            phones = ['sil'] + phones
        if phones[-1] != 'sil':
            phones.append('sil')

        return self.forward(phones) # this is the phone id in self.lexicon
    
    def text_to_jyutpings(self, text):
        """
        Convert text into jyutpings.
        :param text: str, input text.
        :param silence: bool, whether to include silence tokens.
        :return: list of str, phonemes
        """
       
        jyutpings, _ = preprocess_canton(
                text, self.lexicon, silence=False)

        return jyutpings

    def text_to_pinyins(self, text):
        """
        把普通话文本转成带声调的拼音音节，供 MFA 的 .lab 使用。

        标点、英文和词典外字符会被丢弃（不做静音填充），
        静音位置交给 MFA 自己判断。
        :param text: str, input text.
        :return: list of str, 拼音音节（如 jin1 tian1）
        """
        normalized = normalize_text(text)
        pinyins = [
            p[0]
            for p in pinyin(
                normalized, style=Style.TONE3,
                strict=False, neutral_tone_with_five=True
            )
        ]
        return [p for p in pinyins if p in self.lexicon]
