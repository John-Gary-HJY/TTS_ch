"""acoustic model: inference from single utterance info"""
import argparse
import logging
import os

import numpy as np
import torch

import utils
from duritaco.unidirectional_DuriTaco import Unidirectional_DuriTaco as TTSModel
from text.text_frontend import TextFrontend

import json
from hifigan_vocoder.models import Generator
from hifigan_vocoder.env import AttrDict
from scipy.io import wavfile

from duritaco.common_module import Duration_Predictor as DPModel



INT16_MAX=2**15-1

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', default='./train_files', 
    help='Directory containing training files')
parser.add_argument(
    '--model_dir', default='./experiments/acoustic_model_demo')
parser.add_argument(
    '--dp_model_dir', default='./experiments/duration_predictor_demo')
parser.add_argument(
    '--restore_file', help='path to the acoustic model checkpoint to restore')
parser.add_argument(
    '--dp_restore_file', help='path to the duration predictor checkpoint to restore')
parser.add_argument(
    '--vocoder_dir', default='./experiments/pretrained_hifigan_vocoder/UNIVERSAL_V1')
parser.add_argument('--vocoder_cp', default='g_02600000')
parser.add_argument('--lan', default='yue', choices=['yue', 'en', 'zh'])
parser.add_argument('--fpath', required=True, help='path to the test_script')
parser.add_argument('--speaker', required=True, help='test speaker ID')


def load_pretrained_hifigan_vocoder(vocoder_dir, checkpoint='g_02500000'):
    config_file = os.path.join(vocoder_dir, 'config.json')
    checkpoint_path = os.path.join(vocoder_dir, checkpoint)
    with open(config_file) as f:
        data = f.read()
    hifigan_hparams = json.loads(data)
    hifigan_hparams = AttrDict(hifigan_hparams)
    generator = Generator(hifigan_hparams).cuda()
    checkpoint_dict = torch.load(checkpoint_path, map_location='cuda')
    state_dict_g = checkpoint_dict
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return hifigan_hparams, generator


def generate_audio(mel, generator):
    audio = generator(mel)
    audio = audio.squeeze()
    audio = audio * INT16_MAX
    audio = audio.detach().cpu().numpy().astype('int16')
    return audio


def predict_phone_dur(dp_model, phone_seq, speaker_id):
    input_length = torch.LongTensor([len(phone_seq)]).cuda()
    phone_seq = torch.LongTensor([phone_seq]).cuda()
    speaker_id = torch.LongTensor([speaker_id]).cuda()
    inputs = phone_seq, input_length, speaker_id
    dp_model.eval()
    with torch.no_grad():
        phone_dur_seq = dp_model.inference(inputs)
    phone_dur_seq = phone_dur_seq.long().data.cpu()
    return phone_dur_seq


def load_script(speaker, fpath="./test_script.txt"):
    with open(fpath) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    text_holder = []
    for i, line in enumerate(lines):
        index = speaker + '_' + str(i+1)
        text = line
        text_holder.append({"index": index, "text": text})
    return text_holder


def gen_save_audio(text, model, dp_model, generator, speaker_id, audiopath, duration_seq=None):
    phone_seq = text_frontend.text_to_phonemes(text)
    input_length = torch.LongTensor([len(phone_seq)]).cuda()
    phone_seq = torch.LongTensor([phone_seq]).cuda()
    speaker_id = torch.LongTensor([speaker_id]).cuda()
    speaker_id_dur = speaker_id
    inputs = phone_seq, input_length, speaker_id_dur

    with torch.no_grad():
        if duration_seq is None:
            phone_dur_seq = dp_model.inference(inputs)
        else:
            phone_dur_seq = duration_seq.cuda()

        phone_dur_seq = phone_dur_seq.long()
        output_length = torch.sum(phone_dur_seq, dim=1).long()
        utt_info = phone_seq, input_length, phone_dur_seq, output_length, speaker_id
        mel_output = model.inference(utt_info)
        audio_target = generate_audio(mel_output, generator)
        wavfile.write(audiopath, hifigan_hparams.sampling_rate, audio_target)
        logging.info("audiopath: {}".format(audiopath))
        return audio_target



if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'hparams.json')
    dp_json_path = os.path.join(args.dp_model_dir, 'hparams.json')
    fpath = args.fpath
    test_speaker = args.speaker
    
    params = utils.Params(json_path)
    dp_params = utils.Params(dp_json_path)

    text_frontend = TextFrontend(lan=args.lan)
    speaker_map = {}
    with open(os.path.join(args.data_dir, 'speaker_list.txt')) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        speaker_map[line.strip()] = i

    n_symbols = len(text_frontend.SYMBOLS)
    n_spks = len(speaker_map)
    params.n_symbols = n_symbols
    params.n_speakers = n_spks
    dp_params.n_symbols = n_symbols
    dp_params.n_speakers = n_spks
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'testing.log'))

    model = TTSModel(params).cuda()

    restore_path = os.path.join(args.model_dir, args.restore_file)
    logging.info("Restoring parameters from {}".format(restore_path))
    try:
        checkpoint = utils.load_checkpoint(restore_path, model)
    except:
        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint)

    logging.info("Loading the generator")
    hifigan_hparams, generator = load_pretrained_hifigan_vocoder(args.vocoder_dir, args.vocoder_cp)

    # Test with predicted phone duration
    dp_model = DPModel(dp_params).cuda()
    dp_restore_path = os.path.join(args.dp_model_dir, args.dp_restore_file)
    logging.info("Restoring parameters from {} for duration predictor".format(dp_restore_path))
    try:
        checkpoint = utils.load_checkpoint(dp_restore_path, dp_model)
    except:
        checkpoint = torch.load(dp_restore_path)
        dp_model.load_state_dict(checkpoint)


    dp_model.eval()
    model.eval()
    outputname = os.path.basename(fpath).replace(".txt", "")
   

    outputdir = os.path.join(args.model_dir, "outputs_epoch_{}_{}".format(args.restore_file, outputname))
    os.makedirs(outputdir, exist_ok=True)

    logging.info("Start synthesising...")

    test_script = load_script(fpath=fpath, speaker=test_speaker)

    for i, testinfo in enumerate(test_script):
        index = testinfo["index"]
        speaker_name = index.split('_')[0]
        speaker_id = speaker_map[speaker_name]
        text = testinfo["text"]
        audio_name = "{}.wav".format(index)
        audiopath = os.path.join(outputdir, audio_name)
    
        audio = gen_save_audio(text, model, dp_model, generator, speaker_id, audiopath)

 




