import torch
import torch.nn as nn


class ShortTermEncoder(nn.Module):
    """BiLSTM short-term encoder producing a fixed embedding per sequence.

    Input: (batch, seq_len, input_dim)
    Output: (batch, output_dim)
    """
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=128, dropout=0.3, num_layers=2):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        if x is None:
            raise ValueError('Input tensor x is None')
        if x.dim() != 3:
            raise ValueError(f'Expected 3D input (batch, seq_len, input_dim), got {x.dim()}D')
        if x.size(-1) != self.lstm.input_size:
            raise ValueError(f'Input feature dim {x.size(-1)} does not match expected {self.lstm.input_size}')
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.norm(last)
        return self.relu(self.fc(last))
