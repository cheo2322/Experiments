import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads=4, n_layers=2, ff_dim=256, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, src):
        embedded = self.embedding(src)  # [batch, seq, emb_dim]
        # Máscara de padding: True donde hay <pad>
        src_key_padding_mask = (src == self.pad_idx)
        memory = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads=4, n_layers=2, ff_dim=256, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, trg, memory, memory_key_padding_mask=None):
        embedded = self.embedding(trg)  # [batch, seq, emb_dim]

        # Máscara causal: evita que el decoder vea tokens futuros
        seq_len = trg.size(1)
        float_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(trg.device)
        tgt_mask = float_mask == float('-inf')  # convertir a bool

        # Máscara de padding para target
        tgt_key_padding_mask = (trg == self.pad_idx)

        output = self.transformer(
            embedded, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        logits = self.fc(output)  # [batch, seq, vocab]
        return logits


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # Encoder produce memory y su máscara de padding
        memory, src_key_padding_mask = self.encoder(src)
        # Decoder usa memory + secuencia target (teacher forcing)
        outputs = self.decoder(trg, memory, memory_key_padding_mask=src_key_padding_mask)
        return outputs
