import torch
import torch.nn as nn
import numpy as np
import librosa
import json

from text.text_frontend import TextFrontend
from hifigan_vocoder.meldataset import mel_spectrogram

import os


def str_to_int_list(s):
    return list(map(int, s.split()))


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    """
    Your dataset should be of the following structure: wavs and their text filelist with transcriptions.
    In filelist be sure that each line consists: text, phonemes start, phonemes duration, phonemes, wav filename, speaker
    separated with "|".
    """

    def __init__(self, datadir, filelist, n_fft=1024, n_mels=80,
                 sample_rate=22050, hop_length=256, win_length=1024,
                 f_min=0., f_max=8000, load_mels_from_disk=True, lan='en'):
        super(TextMelSpeakerDataset, self).__init__()

        filelist = os.path.join(datadir, filelist)
        with open(filelist, 'r') as f:
            self._metadata = [line.replace('\n', '') for line in f.readlines()]

        with open(os.path.join(datadir, 'speaker_list.txt')) as f:
            lines = f.readlines()
            
        self.speaker_map = dict()
        for i, line in enumerate(lines):
            self.speaker_map[line.strip()] = i

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max

        self._load_mels_from_disk = load_mels_from_disk

        self.text_frontend = TextFrontend(lan=lan)


    def _get_mel(self, filename):
        if self._load_mels_from_disk:
            return torch.from_numpy(np.load(filename))

        audio, _ = librosa.load(
            filename, sr=self.sampling_rate, dtype=np.float32)
        audio = torch.tensor(audio).unsqueeze(0)
        melspec = mel_spectrogram(
            audio,
            n_fft=self.n_fft,
            num_mels=self.n_mels,
            sampling_rate=self.sample_rate,
            hop_size=self.hop_length,
            win_size=self.win_length,
            fmin=self.f_min,
            fmax=self.f_max)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        item_meta = self._metadata[index]
        text, phonemes_start, phonemes_duration, phonemes_code, filename, speaker = item_meta.split(
            '|')
        # Converting all phoneme representations into embedding ids
        phoneme_list = phonemes_code.split()
        phonemes_code = self.text_frontend.forward(phonemes_code.split())
        phonemes_code = torch.tensor(phonemes_code).long()

        item = {
            'text': text,
            'phonemes_start': str_to_int_list(phonemes_start),
            'phonemes_duration': str_to_int_list(phonemes_duration),
            'phonemes_code': phonemes_code,
            'mel': self._get_mel(filename),
            'speaker_id': self.speaker_map[speaker],
            'speaker_name': speaker,
            'index': filename,
            'phoneme_list': phoneme_list
        }

        return item

    def __len__(self):
        return len(self._metadata)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextSpeakerDataset(torch.utils.data.Dataset):
    """
    Your dataset should be of the following structure: wavs and their text filelist with transcriptions.
    In filelist be sure that each line consists: text, phonemes start, phonemes duration, phonemes, wav filename, speaker
    separated with "|".
    """

    def __init__(self, datadir, filelist, lan='en'):
        super(TextSpeakerDataset, self).__init__()

        filelist = os.path.join(datadir, filelist)
        with open(filelist, 'r') as f:
            self._metadata = [line.replace('\n', '') for line in f.readlines()]

        with open(os.path.join(datadir, 'speaker_list.txt')) as f:
            lines = f.readlines()
            
        self.speaker_map = dict()
        for i, line in enumerate(lines):
            self.speaker_map[line.strip()] = i

        self.text_frontend = TextFrontend(lan=lan)


    def __getitem__(self, index):
        item_meta = self._metadata[index]
        text, phonemes_start, phonemes_duration, phonemes_code, filename, speaker = item_meta.split(
            '|')
        # Converting all phoneme representations into embedding ids
        phoneme_list = phonemes_code.split()
        phonemes_code = self.text_frontend.forward(phonemes_code.split())
        phonemes_code = torch.tensor(phonemes_code).long()

        item = {
            'text': text,
            'phonemes_start': str_to_int_list(phonemes_start),
            'phonemes_duration': str_to_int_list(phonemes_duration),
            'phonemes_code': phonemes_code,
            'speaker_id': self.speaker_map[speaker],
            'speaker_name': speaker,
            'index': filename,
            'phoneme_list': phoneme_list
        }

        return item

    def __len__(self):
        return len(self._metadata)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class BatchCollate(object):
    """
    Collates batch objects with padding, decreasing sort by input length, etc.
    """

    def __call__(self, batch):
        B = len(batch)
        # Sorting batch by length of inputs
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['phonemes_code']) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        # Get max mel length
        max_target_len = max([x['mel'].shape[1] for x in batch])
        n_mel = batch[0]['mel'].size(0)

        sequences_padded = torch.zeros(B, max_input_len, dtype=torch.long)
        durations_padded = torch.zeros(B, max_input_len, dtype=torch.long)
        mels_padded = torch.zeros(
            B, n_mel, max_target_len, dtype=torch.float32)

        output_lengths = torch.zeros(B).long()
        speaker_ids = torch.zeros(B).long()

        for index, i in enumerate(ids_sorted_decreasing):
            x = batch[i]

            sequence = torch.LongTensor(x['phonemes_code'])
            sequences_padded[index, :len(
                sequence)] = sequence
            duration = torch.LongTensor(x['phonemes_duration'])
            durations_padded[index, :len(
                duration)] = duration
            mel = torch.FloatTensor(x['mel'])
            # B, 80, T
            mels_padded[index, :, :mel.shape[1]] = mel

            output_lengths[index] = mel.shape[1]
            speaker_ids[index] = x['speaker_id']

        x = (sequences_padded, input_lengths, durations_padded, output_lengths, speaker_ids)
        y = mels_padded

        return x, y


class BatchCollate_Wo_Mel(object):
    """
    Collates batch objects with padding, decreasing sort by input length, etc.
    """

    def __call__(self, batch):
        B = len(batch)
        # Sorting batch by length of inputs
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['phonemes_code']) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]


        sequences_padded = torch.zeros(B, max_input_len, dtype=torch.long)
        durations_padded = torch.zeros(B, max_input_len, dtype=torch.long)
      
        speaker_ids = torch.zeros(B).long()

        for index, i in enumerate(ids_sorted_decreasing):
            x = batch[i]

            sequence = torch.LongTensor(x['phonemes_code'])
            sequences_padded[index, :len(
                sequence)] = sequence
            duration = torch.LongTensor(x['phonemes_duration'])
            durations_padded[index, :len(
                duration)] = duration

            speaker_ids[index] = x['speaker_id']

        x = (sequences_padded, input_lengths, durations_padded, speaker_ids)

        return x


def batch_to_gpu(batch):
    x, y = batch
    phone_seq_padded, input_lengths, phone_dur_seq, \
        output_lengths, speaker_ids = x
    mel_padded = y
    phone_seq_padded = phone_seq_padded.cuda()
    input_lengths = input_lengths.cuda()
    phone_dur_seq = phone_dur_seq.cuda()
    mel_padded = mel_padded.cuda()
    output_lengths = output_lengths.cuda()
    speaker_ids = speaker_ids.cuda()
    x = (phone_seq_padded, input_lengths,
         phone_dur_seq, output_lengths, speaker_ids)
    y = mel_padded

    return x, y

