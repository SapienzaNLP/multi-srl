import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class SequenceEncoder(nn.Module):

    def __init__(
        self,
        encoder_type='lstm',
        input_size=512,
        hidden_size=512,
        num_layers=2,
        dropout=0.1,
    ):
        super(SequenceEncoder, self).__init__()
        if encoder_type == 'lstm':
            self.sequence_encoder = StackedBiLSTM(
                lstm_input_size=input_size,
                lstm_hidden_size=hidden_size,
                lstm_num_layers=num_layers,
                lstm_dropout=dropout,
            )
        elif encoder_type == 'connected_lstm':
            self.sequence_encoder = FullyConnectedBiLSTM(
                lstm_input_size=input_size,
                lstm_hidden_size=hidden_size,
                lstm_num_layers=num_layers,
                lstm_dropout=dropout,
            )
        elif encoder_type == 'residual_lstm':
            self.sequence_encoder = ResidualBiLSTM(
                lstm_input_size=input_size,
                lstm_hidden_size=hidden_size,
                lstm_num_layers=num_layers,
                lstm_dropout=dropout,
            )
        else:
            raise NotImplementedError('{} is not implemented.'.format(self.encoder_type))

        self.output_size = self.sequence_encoder.output_size

    def forward(self, input_sequences, sequence_lengths=None):
        return self.sequence_encoder(input_sequences, sequence_lengths)


class StackedBiLSTM(nn.Module):

    def __init__(
        self,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_dropout
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(lstm_input_size)

        self.lstm = nn.LSTM(
            lstm_input_size,
            lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True)

        self.output_size = 2 * lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):

        total_length = input_sequences.shape[1]
        input_sequences = self.layer_norm(input_sequences)

        packed_input = pack_padded_sequence(
            input_sequences,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False)

        packed_sequence_encodings, _ = self.lstm(packed_input)

        sequence_encodings, _ = pad_packed_sequence(
            packed_sequence_encodings,
            total_length=total_length,
            batch_first=True)

        return sequence_encodings


class FullyConnectedBiLSTM(nn.Module):

    def __init__(
        self,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_dropout
    ):
        super().__init__()

        _lstms = []
        _norms = []
        _drops = []
        _layer_input_size = lstm_input_size

        self.input_normalization = nn.LayerNorm(lstm_input_size)

        for _ in range(lstm_num_layers):
            lstm = nn.LSTM(
                _layer_input_size,
                lstm_hidden_size,
                bidirectional=True,
                batch_first=True)

            norm = nn.LayerNorm(2*lstm_hidden_size)
            drop = nn.Dropout(lstm_dropout)

            _layer_input_size += 2*lstm_hidden_size

            _lstms.append(lstm)
            _norms.append(norm)
            _drops.append(drop)

        self.lstms = nn.ModuleList(_lstms)
        self.norms = nn.ModuleList(_norms)
        self.drops = nn.ModuleList(_drops)

        self.output_size = lstm_input_size + lstm_num_layers*(2*lstm_hidden_size)

    def forward(self, input_sequences, sequence_lengths):

        total_length = input_sequences.shape[1]
        input_sequences = self.input_normalization(input_sequences)

        for lstm, drop, norm in zip(self.lstms, self.drops, self.norms):
            packed_input = pack_padded_sequence(
                input_sequences,
                sequence_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False)

            packed_sequence_encodings, _ = lstm(packed_input)

            sequence_encodings, _ = pad_packed_sequence(
                packed_sequence_encodings,
                total_length=total_length,
                batch_first=True)

            sequence_encodings = norm(sequence_encodings)
            sequence_encodings = drop(sequence_encodings)
            input_sequences = torch.cat([input_sequences, sequence_encodings], dim=-1)
        
        output_sequences = input_sequences

        return output_sequences


class ResidualBiLSTM(nn.Module):

    def __init__(
        self,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_dropout
    ):
        super().__init__()

        self.input_projection = nn.Linear(lstm_input_size, 2*lstm_hidden_size)
        self.input_normalization = nn.LayerNorm(2*lstm_hidden_size)

        _lstms = []
        _norms = []
        _drops = []
        _layer_input_size = 2*lstm_hidden_size

        for _ in range(lstm_num_layers):
            lstm = nn.LSTM(
                _layer_input_size,
                lstm_hidden_size,
                bidirectional=True,
                batch_first=True)

            norm = nn.LayerNorm(2*lstm_hidden_size)
            drop = nn.Dropout(lstm_dropout)

            _layer_input_size = 2*lstm_hidden_size
            _lstms.append(lstm)
            _norms.append(norm)
            _drops.append(drop)

        self.lstms = nn.ModuleList(_lstms)
        self.norms = nn.ModuleList(_norms)
        self.drops = nn.ModuleList(_drops)

        self.output_size = 2*lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):

        total_length = input_sequences.shape[1]
        input_sequences = self.input_projection(input_sequences)
        input_sequences = self.input_normalization(input_sequences)

        for lstm, drop, norm in zip(self.lstms, self.drops, self.norms):

            packed_input = pack_padded_sequence(
                input_sequences,
                sequence_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False)

            packed_sequence_encodings, _ = lstm(packed_input)

            sequence_encodings, _ = pad_packed_sequence(
                packed_sequence_encodings,
                total_length=total_length,
                batch_first=True)

            sequence_encodings = drop(sequence_encodings)
            sequence_encodings = norm(sequence_encodings)
            sequence_encodings = sequence_encodings + input_sequences
            input_sequences = sequence_encodings

        output_sequences = input_sequences
        return output_sequences