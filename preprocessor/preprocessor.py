import os
import random
import json
import math

import textgrid
import librosa
import numpy as np
import torch
from tqdm import tqdm
import random

from hifigan_vocoder.meldataset import mel_spectrogram


class Preprocessor:
    def __init__(self, config):
        self.config = config
        # Speaker
        self.speakers = config["speakers"]
        # Path
        self.audio_dir = config["path"]["audio_path"]
        self.text_dir = config["path"]["text_path"]
        self.tgt_dir = config["path"]["textgrid_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.file_out_dir = config["path"]["train_files_path"]
        # train-test split
        if "val_size" in config["preprocessing"]:
            self.val_size = config["preprocessing"]["val_size"]
        # audio preprocessing parameters
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.filter_length = config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.win_length = config["preprocessing"]["stft"]["win_length"]
        self.n_mel_channels = config["preprocessing"]["mel"]["n_mel_channels"]
        self.mel_fmin = config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = config["preprocessing"]["mel"]["mel_fmax"]

    def build_from_path(self):
        print("Processing Data ...")
        out = list()
        total_time = {}

        # Compute duration, and mel-spectrogram
        for speaker in self.speakers:
            n_frames = 0
            total_time[speaker] = 0

            os.makedirs(
                (os.path.join(self.out_dir, "mel", speaker)), exist_ok=True)
            
            for wav_name in os.listdir(os.path.join(self.audio_dir, speaker)):
                if wav_name[-4:] not in [".WAV", ".wav"]:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.tgt_dir, speaker, "{}.TextGrid".format(basename)
                )
                print(tg_path)
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, n = ret
                    out.append(info)

                n_frames += n

            total_time[speaker] = n_frames * self.hop_length / self.sampling_rate

        for speaker in self.speakers:
            print(
                "Total time of {}: {} hours".format(
                    speaker, total_time[speaker] / 3600
                )
            )

        #shuffle the list out
        random.shuffle(out)

        # Write metadata
        with open(os.path.join(self.file_out_dir, "train_{}.txt".format("_".join(self.speakers))), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")
        with open(os.path.join(self.file_out_dir, "val_{}.txt".format("_".join(self.speakers))), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        # Write train files list
        with open(os.path.join(self.file_out_dir, "train_{}.txt".format("_".join(self.speakers))), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        def audio_amp_normalization(audio_data):
            scale = random.uniform(0.90, 0.95)
            scale = scale * 1.0 / max(abs(audio_data))
            return audio_data * scale

        wav_path = os.path.join(self.audio_dir, speaker,
                                "{}.wav".format(basename))
        if not os.path.exists(wav_path):
            wav_path = wav_path.replace(".wav", ".WAV")

        text_path = os.path.join(self.text_dir, speaker,
                                 "{}.lab".format(basename))
        tg_path = os.path.join(
            self.tgt_dir, speaker, "{}.TextGrid".format(basename))

        # Get alignments
        tg = textgrid.TextGrid.fromFile(tg_path)
        phone_tier = tg.getFirst("phones")  # or tg.getList("phones")[0]
        phone, durations, stamps = self.get_alignment(phone_tier)
        text = " ".join(phone)

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        ### filter out too long or too short audio
        if wav.shape[0] > int(self.sampling_rate*25.0) or wav.shape[0] < int(self.sampling_rate*1.0):
            return None
        wav = audio_amp_normalization(wav)
        wav = wav.astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n").lower()

        # Compute mel-scale spectrogram
        print(tg_path)
        mel_spectrogram = self.get_mel(wav)

        if abs(mel_spectrogram.shape[1] - sum(durations)) > 2:
            print(
            "Discarding sample {} because mel frames {} != durations {}".format(
                basename, mel_spectrogram.shape[1], sum(durations)
            )
            )
            return None

        mel_spectrogram = mel_spectrogram[:, : sum(durations)]
        if mel_spectrogram.shape[1] != sum(durations):
            return None

        mel_filename = "{}.npy".format(basename)
        mel_savepath = os.path.join(self.out_dir, "mel", speaker, mel_filename)
        np.save(
            mel_savepath,
            mel_spectrogram,
        )

        stamps_str = " ".join([str(stamp) for stamp in stamps])
        durations_str = " ".join([str(duration) for duration in durations])

        return (
            "|".join([raw_text, stamps_str, durations_str,
                     text, mel_savepath, speaker]),
            mel_spectrogram.shape[1],
        )


    def get_alignment(self, tier):
        def time2frame(t):
            return math.floor((t+1e-9) * self.sampling_rate / self.hop_length)

        sil_phones = ["", "sp", "spn"]

        phones = []
        stamps = []
        durations = []

        for interval in tier:
            s = interval.minTime
            e = interval.maxTime
            p = interval.mark  # instead of .label

            if p not in sil_phones:
                phones.append(p)
            else:
                phones.append('sil')

            start_frame = time2frame(s)
            end_frame = max(time2frame(e), start_frame+1)

            stamps.append(start_frame)
            durations.append(end_frame-start_frame)

        return phones, durations, stamps

    def get_mel(self, audio):
        audio = torch.tensor(audio).unsqueeze(0)
        melspec = mel_spectrogram(
            audio, n_fft=self.filter_length, num_mels=self.n_mel_channels, sampling_rate=self.sampling_rate, hop_size=self.hop_length, win_size=self.win_length, fmin=self.mel_fmin, fmax=self.mel_fmax)
        melspec = torch.squeeze(melspec, 0).numpy()
        # melshape = 80*T
        return melspec
