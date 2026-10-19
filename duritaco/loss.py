from torch import nn
import torch


class DuriTacoLoss(nn.Module):
    def __init__(self):
        super(DuriTacoLoss, self).__init__()

    def forward(self, mel_outputs, mel_target, mask=None):
        # mel_target of shape ( B, C, T)
        mel_target.requires_grad = False
        if mask is not None:
            mask = mask.unsqueeze(1)
            mel_outputs = mel_outputs.masked_select(mask)
            mel_target = mel_target.masked_select(mask)
        mel_loss = nn.MSELoss(reduction='mean')(mel_outputs, mel_target)
        # mel_loss = nn.L1Loss(reduction='mean')(mel_outputs, mel_target)
        return mel_loss
    

class WeightedDuriTacoLoss(nn.Module):
    def __init__(self):
        super(WeightedDuriTacoLoss, self).__init__()

    def forward(self, mel_outputs, mel_target, mask=None, weights=None):
        # mel_target of shape ( B, C, T)
        mel_target.requires_grad = False
        mel_loss = (mel_outputs - mel_target)**2
        mel_loss = torch.sum(mel_loss, dim=1)
        mel_loss = mel_loss.masked_select(mask)
        # weights = torch.float(mask) * weights
        # weights = weights/torch.sum(weights, dim=1, keepdim=True)
        weights = weights.masked_select(mask)
        # print("mask", mask.shape, "mel_loss", mel_loss.shape, "weights", weights.shape)
        mel_loss = torch.log(mel_loss * weights)
        return torch.sum(mel_loss)/len(mel_loss)

class Duration_Loss(nn.Module):
    def __init__(self):
        super(Duration_Loss, self).__init__()
      
    def forward(self, duration_outputs, phone_dur_seq, input_lengths, mask=None):
        duration_targets = duration_outputs.new(duration_outputs.size()).zero_()
        # compute the log duration targets
        for i in range(duration_targets.size(0)):
            duration_targets[i] \
                = torch.log((phone_dur_seq[i]+1.0).float()).to(duration_outputs.device)
        if mask is not None:
            duration_outputs = duration_outputs.masked_select(mask)
            duration_targets = duration_targets.masked_select(mask)
        duration_loss = nn.MSELoss()(duration_outputs, duration_targets)
        return duration_loss