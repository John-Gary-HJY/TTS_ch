"""train the acoustic model"""
import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data import BatchCollate_Wo_Mel, TextSpeakerDataset
from duritaco.common_module import Duration_Predictor as DPModel
from duritaco.loss import Duration_Loss
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', default='./train_files', 
    help='Directory containing the training files')
parser.add_argument(
    '--model_dir', default='./experiments/duration_predictor_demo', 
    help='Directory containing params.json')
parser.add_argument(
    '--restore_file', default=None, help='Optional, name of the file in\
         --model_dir containing weights to reload before training')
parser.add_argument(
    '--training_file', default='train.txt')
parser.add_argument(
    '--validation_file', default='val.txt')
parser.add_argument(
    '--lan', default='yue', choices=['yue', 'en', 'zh'])

def train(model, optimizer, loss_fn, data_loader, params):
    model.train()
    summ = []
    loss_avg = utils.RunningAverage()
    tepoch = tqdm(enumerate(data_loader), total=len(data_loader))

    for i, batch in tepoch:
        phone_seq_padded, input_lengths, phone_dur_seq, speaker_ids = batch

        if params.cuda:
            phone_seq_padded = phone_seq_padded.cuda()
            input_lengths = input_lengths.cuda()
            speaker_ids = speaker_ids.cuda()

        inputs = (phone_seq_padded, input_lengths, speaker_ids)

        duration_outputs, output_mask = model(inputs)
        
        loss = loss_fn(duration_outputs, phone_dur_seq, input_lengths, output_mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            summary_batch={'loss': loss.item()}
            summ.append(summary_batch)
        
        # update the average loss
        loss_avg.update(loss.item())
        tepoch.set_postfix(loss='{:05.3f}'.format(loss_avg()))
        # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

def evaluate(model, data_loader):
    model.eval()
    summ = []
    for _, batch in enumerate(data_loader):
        phone_seq_padded, input_lengths, phone_dur_seq, speaker_ids = batch
        if params.cuda:
            phone_seq_padded = phone_seq_padded.cuda()
            input_lengths = input_lengths.cuda()
            speaker_ids = speaker_ids.cuda()

        inputs = (phone_seq_padded, input_lengths, speaker_ids)
        duration_predict = model.inference(inputs).data.cpu().numpy()
        duration_target = np.array(phone_dur_seq[0])
        frame_diff = np.sum(abs(duration_predict - duration_target))/float(np.sum(duration_target))
        summ.append({"frame_error": frame_diff})
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)



def train_and_evaluate(model, data_loader, data_loader_val, optimizer, loss_fn, params, model_dir, restore_file=None):
    last_epoch = 0
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['state_dict'])
        last_epoch = checkpoint['epoch']

    for epoch in range(last_epoch+1, params.num_epochs+1):
        logging.info("Epoch {}/{}".format(epoch, params.num_epochs))
        train(model, optimizer, loss_fn, data_loader, params )
        evaluate(model, data_loader_val)

        if epoch % params.iters_per_checkpoint == 0:
            utils.save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()}, checkpoint=model_dir, name=str(epoch))

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'hparams.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    
    params = utils.Params(json_path)
    # use GPU if available
    params.cuda = torch.cuda.is_available()
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(412)
    if params.cuda: torch.cuda.manual_seed(412)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    dataset = TextSpeakerDataset(
        args.data_dir, args.training_file, lan=args.lan)
    dataset_val = TextSpeakerDataset(
        args.data_dir, args.validation_file, lan=args.lan)
    logging.info("- done.")
    logging.info("{} training samples in total.".format(len(dataset)))

    collate_fn = BatchCollate_Wo_Mel()
    data_loader = DataLoader(dataset, num_workers=4, shuffle=True,
                             batch_size=params.batch_size, collate_fn=collate_fn, drop_last=False)
    data_loader_val = DataLoader(dataset_val, shuffle=False,
                             batch_size=1, collate_fn=collate_fn, drop_last=False)

    n_symbols = len(dataset.text_frontend.SYMBOLS)
    n_spks = len(dataset.speaker_map)
    params.n_symbols = n_symbols
    params.n_speakers = n_spks


    model = DPModel(params).cuda() if params.cuda else DPModel(params)
    loss_fn = Duration_Loss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

        
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, data_loader, data_loader_val, optimizer, loss_fn, params, args.model_dir, args.restore_file)
   