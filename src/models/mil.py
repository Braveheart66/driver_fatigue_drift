import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning (MIL) wrapper.

    Wraps a pretrained short-term encoder.
    Bag = one session.
    Instances = sliding context windows within the session.

    The attention weights reveal which moments in time drove the session-level label,
    providing temporal attribution.
    """
    def __init__(self, encoder, embedding_dim=128):
        super().__init__()
        self.encoder = encoder
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, bag):
        # bag: (n_instances, seq_len, input_dim)
        if bag.dim() != 3:
            raise ValueError(f"Expected 3D bag input (n_instances, seq_len, input_dim), got {bag.dim()}D")
        
        embeddings = self.encoder(bag)                    # (n_instances, embedding_dim)
        attn_logits = self.attention(embeddings)           # (n_instances, 1)
        attn_weights = F.softmax(attn_logits, dim=0)        # (n_instances, 1)
        
        # Weighted sum: (1, embedding_dim)
        bag_embedding = (attn_weights * embeddings).sum(dim=0, keepdim=True)
        
        # Output logit: (1,)
        logits = self.classifier(bag_embedding).squeeze(-1)
        return logits, attn_weights
