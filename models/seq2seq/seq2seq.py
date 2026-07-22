import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Positional embedding aprendido. Se suma a los embeddings de token."""
    def __init__(self, max_len, emb_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

    def forward(self, x):
        # x: (batch, seq_len, emb_dim)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        return x + self.pos_embedding(positions)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, n_heads=8, n_layers=4, ff_dim=1024,
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

    def forward(self, src):
        embedded = self.embedding(src) * math.sqrt(self.emb_dim)   # <-- fix
        embedded = self.pos_enc(embedded)
    
        src_key_padding_mask = (src == self.pad_idx)
        memory = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)
        
        return memory, src_key_padding_mask


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, n_heads=8, n_layers=4, ff_dim=1024,
                 pad_idx=0, max_len=64):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEmbedding(max_len, emb_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, trg, memory, memory_key_padding_mask=None):
        embedded = self.embedding(trg) * math.sqrt(self.emb_dim)   # <-- fix
        embedded = self.pos_enc(embedded)
        
        seq_len = trg.size(1)
        float_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(trg.device)
        tgt_mask = float_mask == float('-inf')
        tgt_key_padding_mask = (trg == self.pad_idx)
        output = self.transformer(
            embedded, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return self.fc(output)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        memory, src_key_padding_mask = self.encoder(src)

        return self.decoder(trg, memory, memory_key_padding_mask=src_key_padding_mask)
