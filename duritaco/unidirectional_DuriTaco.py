"""This model is a unidirectional version of DuriTaco, which is used in the normal text-to-speech scenario"""

import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt

from duritaco.common_module import Prenet, LinearNorm, Encoder, Position_Encoder, Length_Regulator, \
    get_mask_from_lengths


class Unidirectional_Decoder(nn.Module):
    def __init__(self, hparams, stats=False):
        super(Unidirectional_Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        # encoder output embedding dimension
        self.encoder_embedding_dim = hparams.symbol_embedding_dim \
            + hparams.position_embedding_dim + hparams.speaker_embedding_dim

        if stats==True:
            self.encoder_embedding_dim = hparams.symbol_embedding_dim \
                + hparams.position_embedding_dim

        # module dimension
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        # module
        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])
        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim)
        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)
        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

    def get_go_frame(self, expanded_encoder_outputs):
        """
        Gets all zeros frames to use as first decoder input and encoder output
        """
        batch_size = expanded_encoder_outputs.size(0)
        encoder_embedding_dim = expanded_encoder_outputs.size(2)
        go_decoder_input = expanded_encoder_outputs.new(
            batch_size, self.n_mel_channels * self.n_frames_per_step).zero_()
        go_encoder_output = expanded_encoder_outputs.new(batch_size, encoder_embedding_dim).zero_()
        return go_decoder_input, go_encoder_output

    def initialize_decoder_states(self, batch_size, expanded_encoder_outputs):
        """
        Initializes attention rnn states, decoder rnn states
        """
        self.attention_hidden = expanded_encoder_outputs.new(batch_size, self.attention_rnn_dim).zero_()
        self.attention_cell = expanded_encoder_outputs.new(batch_size, self.attention_rnn_dim).zero_()
        self.decoder_hidden = expanded_encoder_outputs.new(batch_size, self.decoder_rnn_dim).zero_()
        self.decoder_cell = expanded_encoder_outputs.new(batch_size, self.decoder_rnn_dim).zero_()

    def decode(self, preneted_decoder_input, encoder_output_past, encoder_output_current):
        """
        Decoder step using stored states, decoder input and encoder output of past and current step in normal order
        """
        attention_rnn_input = torch.cat([preneted_decoder_input, encoder_output_past], dim=-1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            attention_rnn_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)
        decoder_rnn_input = torch.cat(
            [self.attention_hidden, encoder_output_current], dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_rnn_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)
        decoder_hidden_attention_context = torch.cat(
            [self.decoder_hidden, encoder_output_current], dim=-1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        return decoder_output

    def unidirection_forward(self, go_decoder_input, decoder_inputs, go_encoder_output, expanded_encoder_outputs):
        go_encoder_output = go_encoder_output.unsqueeze(0)  # of shape (1, batch_size, encoder_embedding_dim)
        expanded_encoder_outputs = expanded_encoder_outputs.permute(1, 0, 2)
        # of shape (T_out, batch_size, encoder_embedding_dim)
        expanded_encoder_outputs_with_go_frame = torch.cat([go_encoder_output, expanded_encoder_outputs], dim=0)
        # of shape (T_out+1, batch_size, encoder_embedding_dim)
        go_decoder_input = go_decoder_input.unsqueeze(0)  # of shape (1, batch_size, num_mel)
        decoder_inputs = decoder_inputs.permute(2, 0, 1)  # of shape (T_out, batch_size, num_mel)
        decoder_inputs = torch.cat([go_decoder_input, decoder_inputs], dim=0)
        # of shape (T_out+1, batch_size, num_mel)
        preneted_decoder_inputs = self.prenet(decoder_inputs)
        decoder_outputs = []
        while len(decoder_outputs) < decoder_inputs.size(0) - 1:
            preneted_decoder_input = preneted_decoder_inputs[len(decoder_outputs)]
            encoder_output_past = expanded_encoder_outputs_with_go_frame[len(decoder_outputs)]
            encoder_output_current = expanded_encoder_outputs_with_go_frame[len(decoder_outputs)]
            decoder_output = self.decode(preneted_decoder_input, encoder_output_past, encoder_output_current)
            decoder_outputs += [decoder_output]
        decoder_outputs = torch.stack(decoder_outputs).contiguous().permute(1, 0, 2)  # B, T, C
        return decoder_outputs

    def unidirection_inference(self, go_decoder_input, go_encoder_output, expanded_encoder_outputs):
        go_encoder_output = go_encoder_output.unsqueeze(0)  # of shape (1, batch_size, encoder_embedding_dim)
        expanded_encoder_outputs = expanded_encoder_outputs.permute(1, 0, 2)
        # of shape (T_out, batch_size, encoder_embedding_dim)
        expanded_encoder_outputs_with_go_frame = torch.cat([go_encoder_output, expanded_encoder_outputs], dim=0)
        # of shape (T_out+1, batch_size, encoder_embedding_dim)
        decoder_outputs = []
        decoder_input = go_decoder_input
        while len(decoder_outputs) < expanded_encoder_outputs_with_go_frame.size(0) - 1:
            preneted_decoder_input = self.prenet(decoder_input)
            encoder_output_past = expanded_encoder_outputs_with_go_frame[len(decoder_outputs)]
            encoder_output_current = expanded_encoder_outputs_with_go_frame[len(decoder_outputs) + 1]
            decoder_output = self.decode(preneted_decoder_input, encoder_output_past, encoder_output_current)
            decoder_outputs += [decoder_output]
            decoder_input = decoder_outputs[-1]
        decoder_outputs = torch.stack(decoder_outputs).contiguous().permute(1, 0, 2)  # B, T, C
        return decoder_outputs

    def forward(self, expanded_encoder_outputs, decoder_inputs):
        batch_size = expanded_encoder_outputs.size(0)
        self.initialize_decoder_states(batch_size, expanded_encoder_outputs)
        go_decoder_input, go_encoder_output = self.get_go_frame(expanded_encoder_outputs)
        decoder_outputs = self.unidirection_forward(go_decoder_input, decoder_inputs,
                                                    go_encoder_output, expanded_encoder_outputs)  # B, T, C

        mel_outputs = decoder_outputs.permute(0, 2, 1)  # B, C,T
        return mel_outputs

    def inference(self, expanded_encoder_outputs):
        batch_size = expanded_encoder_outputs.size(0)
        self.initialize_decoder_states(batch_size, expanded_encoder_outputs)
        go_decoder_input, go_encoder_output, = self.get_go_frame(expanded_encoder_outputs)
        decoder_outputs = self.unidirection_inference(go_decoder_input, go_encoder_output,
                                                      expanded_encoder_outputs)  # B, T, C
        mel_outputs = decoder_outputs.permute(0, 2, 1)
        return mel_outputs


class Unidirectional_DuriTaco(nn.Module):
    def __init__(self, hparams):
        super(Unidirectional_DuriTaco, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_symbols = hparams.n_symbols
        self.n_speakers = hparams.n_speakers

        self.symbol_embedding_dim = hparams.symbol_embedding_dim
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        self.position_embedding_dim = hparams.position_embedding_dim

        self.symbols_embedding = nn.Embedding(self.n_symbols, self.symbol_embedding_dim, padding_idx=0)
        std = sqrt(2.0 / (self.n_symbols + self.symbol_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.symbols_embedding.weight.data.uniform_(-val, val)

        self.speaker_embedding = nn.Embedding(self.n_speakers, self.speaker_embedding_dim)
        torch.nn.init.xavier_uniform_(self.speaker_embedding.weight)
        
        self.encoder = Encoder(hparams)
        self.length_regulator = Length_Regulator()
        self.position_encoder = Position_Encoder()
        self.decoder = Unidirectional_Decoder(hparams)


    def parse_output(self, outputs, output_lengths=None):
        if output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            outputs[0].masked_fill_(mask, 0.0)
        return outputs


    def forward(self, inputs):
        (phone_seq_padded, input_lengths, \
            phone_dur_seqs, output_lengths, speaker_ids), mel_padded = inputs
       
        input_lengths, output_lengths = input_lengths.detach(), output_lengths.detach()
        max_len = max(input_lengths)

        outputs = []

        # Get symbols encoder outputs
        embedded_inputs = self.symbols_embedding(phone_seq_padded).transpose(1,2)
        encoder_outputs = self.encoder(embedded_inputs, input_lengths)
        outputs.append(encoder_outputs)

        # Get speaker embeddings
        speaker_ids = speaker_ids.unsqueeze(1)
        embedded_speakers = self.speaker_embedding(speaker_ids)
        embedded_speakers = embedded_speakers.expand(-1, max_len, -1)
        outputs.append(embedded_speakers)

        merged_outputs = torch.cat(outputs, -1)
        expanded_encoder_outputs = self.length_regulator(
            (merged_outputs, phone_dur_seqs, input_lengths, output_lengths)
            )
        # add phone relative position encoding
        expanded_encoder_outputs = self.position_encoder(
                expanded_encoder_outputs, phone_dur_seqs, output_lengths)
        
        decoder_inputs = mel_padded
        mel_outputs = self.decoder(
            expanded_encoder_outputs, decoder_inputs)
        # mel_outputs = self.parse_output([mel_outputs], output_lengths)[0]
        mel_mask = get_mask_from_lengths(output_lengths)
        return mel_outputs, mel_mask


    def inference(self, inputs):
        # Batch_size==1
        phone_seq, input_length, \
            phone_dur_seq, output_length, speaker_id= inputs
       
        input_length, output_length = input_length.detach(), output_length.detach()
        max_len = max(input_length)

        outputs = []

        # Get symbols encoder outputs
        embedded_inputs = self.symbols_embedding(phone_seq).transpose(1,2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        outputs.append(encoder_outputs)

        # Get speaker embeddings
        speaker_ids = speaker_id.unsqueeze(1)
        embedded_speakers = self.speaker_embedding(speaker_ids)
        embedded_speakers = embedded_speakers.expand(-1, max_len, -1)
        outputs.append(embedded_speakers)

        merged_outputs = torch.cat(outputs, -1)
        expanded_encoder_outputs = self.length_regulator(
            (merged_outputs, phone_dur_seq,input_length, output_length)
            )

        expanded_encoder_outputs = self.position_encoder(
            expanded_encoder_outputs, phone_dur_seq, output_length)
        
        # print("ENCODER OUTPUT SHAPE:", expanded_encoder_outputs.shape)
        # print(expanded_encoder_outputs[0, :, 255])
        # tmp = expanded_encoder_outputs[0, :, :512]
        # tmp = torch.cov(tmp.T)
        # print(tmp.shape)
        # print(tmp)
        # torch.save(tmp.detach().cpu(), 'covariance.pt')
        
        mel_outputs = self.decoder.inference(expanded_encoder_outputs)
        
        return mel_outputs