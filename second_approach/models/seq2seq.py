import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return outputs, hidden.unsqueeze(0)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_dim, dec_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim + enc_dim, dec_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_dim, vocab_size)
        self.attn = BahdanauAttention(enc_dim, dec_dim)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)  # [B, 1, E]
        context, _ = self.attn(hidden[-1], encoder_outputs)  # [B, enc_dim]
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # [B, 1, E+enc_dim]
        output, hidden = self.rnn(rnn_input, hidden)  # output: [B, 1, H]
        pred = self.fc_out(output.squeeze(1))  # [B, vocab_size]
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5, forced_token_id=None):
        batch_size, trg_len = trg.size(0), trg.size(1)
        vocab_size = self.decoder.fc_out.out_features

        # print("[audit] pesos fc_out para índice 'a':", self.decoder.fc_out.weight[0])

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        if forced_token_id is not None:
            _input = torch.full((batch_size, 1), forced_token_id, dtype=torch.long, device=self.device)
        else:
            _input = trg[:, 0].unsqueeze(1)  # fallback a <sos>
        
        # print("[audit] input inicial (índices):", input.squeeze(1).tolist())
        # print("[audit] hidden inicial (mean):", hidden.mean().item())
        # print("[audit] hidden shape:", hidden.shape)
        # print("[audit] encoder_outputs shape:", encoder_outputs.shape)

        for t in range(1, trg_len):
            output, hidden = self.decoder(_input, hidden, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1).unsqueeze(1)
            _input = trg[:, t].unsqueeze(1) if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
    
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [B, H]
        # encoder_outputs: [B, T, enc_dim]

        src_len = encoder_outputs.size(1)

        # Expand hidden para que tenga shape [B, T, H]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [B, T, H]

        # Concatenar en la última dimensión
        concat = torch.cat((hidden, encoder_outputs), dim=2)  # [B, T, H + enc_dim]

        energy = torch.tanh(self.attn(concat))  # [B, T, dec_dim]
        scores = self.v(energy).squeeze(2)      # [B, T]
        attn_weights = torch.softmax(scores, dim=1)  # [B, T]

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B, 1, enc_dim]
        return context.squeeze(1), attn_weights  # [B, enc_dim], [B, T]
