import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)


class PositionwiseModel(nn.Module):
    """
    Reemplaza el encoder-decoder autorregresivo.
    Predice cada byte de salida directamente a partir del byte de entrada
    en la MISMA posición, sin generación secuencial.
    """
    def __init__(self, vocab_size, emb_dim=128, n_heads=4, n_layers=2, ff_dim=256,
                 pad_idx=0, max_len=64):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEmbedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src) * math.sqrt(self.emb_dim)
        embedded = self.pos_enc(embedded)
        src_key_padding_mask = (src == self.pad_idx)
        memory = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)
        return self.fc(memory)   # (batch, seq_len, vocab_size) — una predicción por posición
    