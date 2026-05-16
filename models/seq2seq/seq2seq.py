import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src, src_lengths):
        # Embedding
        embedded = self.embedding(src)
        # Empaquetar secuencias para manejar longitudes variables
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, trg, hidden):
        embedded = self.embedding(trg)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, trg):
        hidden = self.encoder(src, src_lengths)
        outputs, _ = self.decoder(trg, hidden)
        return outputs
