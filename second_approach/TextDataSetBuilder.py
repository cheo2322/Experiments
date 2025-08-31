import pandas as pd
import torch

class TextDatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vocab = self._build_vocab()
        self.pad_idx = self.vocab["<pad>"]
        self.use_sos_eos = cfg["data"].get("use_sos_eos", False)

    def _build_vocab(self):
        fixed = list(self.cfg["vocab"]["fixed_chars"])
        tokens = self.cfg["vocab"]["add_tokens"]
        vocab = {char: idx for idx, char in enumerate(tokens + fixed)}
        return vocab

    def _tokenize(self, sequence):
        tokens = [self.vocab[c] for c in sequence if c in self.vocab]
        if self.use_sos_eos:
            tokens = [self.vocab["<sos>"]] + tokens + [self.vocab["<eos>"]]
        return tokens

    def _pad_sequences(self, sequences):
        max_len = max(len(seq) for seq in sequences)
        return torch.tensor([
            seq + [self.pad_idx] * (max_len - len(seq)) for seq in sequences
        ])

    def build_tensors(self, csv_path):
        df = pd.read_csv(csv_path)
        if "plain_text" not in df.columns or "caesar_encrypted" not in df.columns:
            raise ValueError("❌ El CSV debe tener columnas 'plain_text' y 'caesar_encrypted'")

        src_seqs = [self._tokenize(text) for text in df["caesar_encrypted"]]
        trg_seqs = [self._tokenize(text) for text in df["plain_text"]]

        src_tensor = self._pad_sequences(src_seqs)
        trg_tensor = self._pad_sequences(trg_seqs)

        return src_tensor, trg_tensor, self.vocab