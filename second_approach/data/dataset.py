import pandas as pd
import torch
from torch.utils.data import Dataset

def encode_tokens(tokens, stoi, max_len):
    ids = [stoi[t] for t in tokens]
    pad_id = stoi["<pad>"]
    ids = ids[:max_len] + [pad_id] * max(0, max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def encode_text(text, vocab, max_len, use_sos_eos=False):
    tokens = list(text)
    if use_sos_eos:
        tokens = ["<sos>"] + tokens + ["<eos>"]
    return encode_tokens(tokens, vocab["stoi"], max_len)

class CaesarDataset(Dataset):
    def __init__(self, csv_path, vocab, max_len, use_sos_eos=False):
        df = pd.read_csv(csv_path)
        self.enc = df["caesar_encrypted"].tolist()
        self.pla = df["plain_text"].tolist()
        self.vocab = vocab
        self.max_len = max_len
        self.use_sos_eos = use_sos_eos

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        src = encode_text(self.enc[idx], self.vocab, self.max_len, self.use_sos_eos)
        trg = encode_text(self.pla[idx], self.vocab, self.max_len, self.use_sos_eos)
        return src, trg