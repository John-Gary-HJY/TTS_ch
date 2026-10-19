import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=lengths.new(max_len).long())
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

def get_mask_from_phone_duration(phone_seq_padded, duration_list):
    # phone_seq_padded: B, Tph
    # duration_list: list of list of integers.
    is_valid_phone = phone_seq_padded!=0
    is_valid_phone = is_valid_phone.float()
    lengths = [sum(duration) for duration in duration_list]
    reps = torch.zeros(
        len(duration_list), phone_seq_padded.size(1)).long()
    
    for i in range(len(duration_list)):
        # print(lengths[i])
        # print(duration_list[i])
        reps[i, :len(duration_list[i])] = torch.tensor(duration_list[i])

    reps = reps.to(phone_seq_padded.device)
    
    max_len = max(lengths)

    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0),
                               dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.long()

    range_ = torch.arange(max_len).to(phone_seq_padded.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    mult = mult.float()
    mask = torch.matmul(mult, is_valid_phone.unsqueeze(-1)).squeeze(-1).bool().unsqueeze(1)

    return mask



class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Encoder(torch.nn.Module):
    """
    Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()
        # config
        self.embedding_dim = hparams.encoder_embedding_dim
        self.kernel_size = hparams.encoder_kernel_size
        self.n_convolutions = hparams.encoder_n_convolutions
        self.n_lstm = hparams.encoder_num_layers
        # layers
        convolutions = []
        for _ in range(self.n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(self.embedding_dim,
                         self.embedding_dim,
                         kernel_size=self.kernel_size, stride=1,
                         padding=int((self.kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(self.embedding_dim,
                            int(self.embedding_dim / 2), self.n_lstm,
                            batch_first=True, bidirectional=True)

    def forward(self, input, input_lengths):
        for conv in self.convolutions:
            input = F.dropout(F.relu(conv(input)), 0.5, self.training)
        input = input.transpose(1, 2)  # shape (B, T, C)
        input = nn.utils.rnn.pack_padded_sequence(
            input, input_lengths.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(input)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        return outputs

    def inference(self, input):
        for conv in self.convolutions:
            input = F.dropout(F.relu(conv(input)), 0.5, self.training)
        input = input.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(input)
        return outputs


class Length_Regulator(torch.nn.Module):
    """
    Length regulator module.
    Adapt from espnet implementation of fastspeech2
    """

    def __init__(self):
        super(Length_Regulator, self).__init__()

    def forward(self, inputs):
        encoder_outputs, phone_dur_seqs, \
            input_lengths, output_lengths = inputs
        batch_size, encoder_embedding_dim \
            = encoder_outputs.size(0), encoder_outputs.size(2)

        max_output_len = torch.max(output_lengths).item()

        expanded_encoder_outputs = encoder_outputs.new(
            batch_size, max_output_len, 
            encoder_embedding_dim).zero_()

        for i in range(batch_size):
            encoder_output = encoder_outputs[i][:input_lengths[i]]
            phone_dur_seq = phone_dur_seqs[i][:input_lengths[i]]
            expanded_encoder_output = torch.repeat_interleave(
                encoder_output, phone_dur_seq, dim=0)
            expanded_encoder_outputs[i, :expanded_encoder_output.size(0), :] = expanded_encoder_output

        return expanded_encoder_outputs


class Position_Encoder(torch.nn.Module):
    """
    Position Encoder module.
    Adapt from DurIAN implementation from https://github.com/ivanvovk/DurIAN
    """

    def __init__(self, position_embedding_dim=1):
        super(Position_Encoder, self).__init__()
        self.position_embedding_dim = position_embedding_dim

    def forward(self, expanded_encoder_outputs, phone_dur_seqs, output_lengths):
        batch_size, max_output_len = expanded_encoder_outputs.size(0), expanded_encoder_outputs.size(1)
        position_embeddings = expanded_encoder_outputs.new(
            batch_size, max_output_len, self.position_embedding_dim).zero_()
        for i in range(batch_size):
            phone_dur_seq = phone_dur_seqs[i]
            position_embedding = torch.cat(
                [torch.linspace(0, 1, steps=phone_dur) for phone_dur in phone_dur_seq])
            # assert position_embedding.size(0) == output_lengths[i]
            position_embeddings[i, :position_embedding.size(0), :] = position_embedding.unsqueeze(dim=-1)
        expanded_encoder_outputs = torch.cat([expanded_encoder_outputs, position_embeddings], dim=-1)
        return expanded_encoder_outputs


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, p_dropout):
        super(ConvReluNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels,
                                                kernel_size, padding=kernel_size//2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(
            torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels, hidden_channels,
                                                    kernel_size, padding=kernel_size//2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask

class Duration_Predictor(nn.Module):
    def __init__(self, hparams):
        super(Duration_Predictor, self).__init__()
        self.n_symbols = hparams.n_symbols
        self.n_speakers = hparams.n_speakers
        self.dropout = hparams.predictor_dropout

        self.symbol_embedding_dim = hparams.symbol_embedding_dim
        self.speaker_embedding_dim = hparams.speaker_embedding_dim

        self.symbols_embedding = nn.Embedding(self.n_symbols, self.symbol_embedding_dim, padding_idx=0)
        std = sqrt(2.0 / (self.n_symbols + self.symbol_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.symbols_embedding.weight.data.uniform_(-val, val)

        self.prenet = ConvReluNorm(self.symbol_embedding_dim, self.symbol_embedding_dim, self.symbol_embedding_dim, kernel_size=5, n_layers=3, p_dropout=0.5)

        self.speaker_embedding = nn.Embedding(self.n_speakers, self.speaker_embedding_dim)
        torch.nn.init.xavier_uniform_(self.speaker_embedding.weight)

        self.duration_rnn_dim = hparams.duration_rnn_dim
        self.duration_rnn_layer = hparams.duration_rnn_layer
        
        self.dropout_layer1= nn.Dropout(self.dropout)

        self.lstm = nn.LSTM(
            input_size=self.symbol_embedding_dim+self.speaker_embedding_dim,
            hidden_size=self.duration_rnn_dim,
            num_layers=self.duration_rnn_layer,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True)

        self.dropout_layer2 = nn.Dropout(self.dropout)

        self.linear_layer = nn.Linear(2*self.duration_rnn_dim, 1)


    def parse_output(self, duration_outputs, output_lengths=None):
        if output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            duration_outputs.masked_fill_(mask, 0.0)
        return duration_outputs

    def forward(self, inputs):
        phone_seq_padded, input_lengths, speaker_ids = inputs
        input_lengths = input_lengths.detach()
        output_mask = get_mask_from_lengths(input_lengths)
        max_len = torch.max(input_lengths)

        outputs = []

        embedded_symbols = self.symbols_embedding(phone_seq_padded)
        embedded_symbols = self.prenet(embedded_symbols.transpose(1,2), output_mask.unsqueeze(1))
        outputs.append(embedded_symbols.transpose(1,2))

        speaker_ids = speaker_ids.unsqueeze(1)
        embedded_speakers = self.speaker_embedding(speaker_ids)

        embedded_speakers = embedded_speakers.expand(-1, max_len, -1)
        outputs.append(embedded_speakers)

        embedded_inputs = torch.cat(outputs, -1)
        embedded_inputs = self.dropout_layer1(embedded_inputs)

        embedded_inputs = nn.utils.rnn.pack_padded_sequence(
            embedded_inputs, input_lengths.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        duration_outputs, _ = self.lstm(embedded_inputs)
        duration_outputs, _ = nn.utils.rnn.pad_packed_sequence(duration_outputs, batch_first=True)

        duration_outputs = self.dropout_layer2(duration_outputs)

        duration_outputs = self.linear_layer(duration_outputs).squeeze(2)
        # duration_outputs = self.parse_output(duration_outputs, input_lengths)
        # output_mask = get_mask_from_lengths(input_lengths)
        return duration_outputs, output_mask

    def inference(self, inputs, len_text=0):
        # batch_size == 1
        log_duration_outputs, _ = self.forward(inputs)
        duration_outputs = torch.clamp(torch.round(1.0*torch.exp(log_duration_outputs)), min=4)
        # if len_text > 10:
        #     duration_outputs = torch.exp(log_duration_outputs)
        #     avg_phone_dur = torch.sum(duration_outputs.squeeze(0))/float(len_text)
        #     duration_outputs = torch.clamp(torch.round(22.0/avg_phone_dur.item() * duration_outputs).long(), min=5)
        # else:
        #     duration_outputs = torch.exp(log_duration_outputs)
        #     avg_phone_dur = torch.sum(
        #         duration_outputs.squeeze(0))/float(len_text)
        #     duration_outputs = torch.clamp(torch.round(
        #         25.0/avg_phone_dur.item() * duration_outputs).long(), min=5)

        return duration_outputs
