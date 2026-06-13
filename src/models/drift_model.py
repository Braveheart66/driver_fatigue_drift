import torch
import torch.nn as nn


class DriftModel(nn.Module):
    """GRU-based long-term drift model.

    Input: (batch, seq_len, embedding_dim)
    Output: fatigue_scores (batch, seq_len, 1), drift_index (batch, 1)
    """
    def __init__(self, embedding_dim=128, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fatigue_out = nn.Linear(hidden_dim, 1)
        self.drift_out = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings):
        # embeddings: (batch, seq, embedding_dim)
        if embeddings is None:
            raise ValueError('Embeddings input is None')
        if embeddings.dim() != 3:
            raise ValueError(f'Expected 3D embeddings (batch, seq, embedding_dim), got {embeddings.dim()}D')
        if embeddings.size(-1) != self.gru.input_size:
            raise ValueError(f'Embedding dim {embeddings.size(-1)} does not match model input size {self.gru.input_size}')
        out, hidden = self.gru(embeddings)
        out = self.dropout(out)
        fatigue_scores = torch.sigmoid(self.fatigue_out(out)) * 100.0
        drift_index = self.drift_out(hidden[-1])
        return fatigue_scores, drift_index
