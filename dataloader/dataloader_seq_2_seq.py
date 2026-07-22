import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import base64

class Seq2SeqDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.plain_texts = df["plain"].tolist()
        self.encrypted_texts = df["encrypted"].tolist()

        # Vocab simple: 256 valores de byte, con <pad>
        self.vocab = ["<pad>"] + [chr(i) for i in range(256)]
        self.char2idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        self.pad_idx = self.char2idx["<pad>"]

    def __len__(self):
        return len(self.plain_texts)

    def encode(self, text):
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def __getitem__(self, idx):
        plain_seq = self.encode(self.plain_texts[idx])

        encrypted_b64 = self.encrypted_texts[idx]
        encrypted_raw = base64.b64decode(encrypted_b64).decode("latin-1")
        encrypted_seq = self.encode(encrypted_raw)

        # Deben tener EXACTAMENTE el mismo largo (RC4 es 1 a 1, byte a byte)
        assert len(plain_seq) == len(encrypted_seq), \
            f"Largo distinto: plain={len(plain_seq)}, encrypted={len(encrypted_seq)}"

        return torch.tensor(encrypted_seq, dtype=torch.long), torch.tensor(plain_seq, dtype=torch.long)


def collate_fn(batch):
    encrypted_seqs, plain_seqs = zip(*batch)
    lengths = [len(seq) for seq in encrypted_seqs]

    encrypted_padded = torch.nn.utils.rnn.pad_sequence(encrypted_seqs, batch_first=True, padding_value=0)
    plain_padded = torch.nn.utils.rnn.pad_sequence(plain_seqs, batch_first=True, padding_value=0)

    return encrypted_padded, plain_padded, lengths, lengths


def get_dataloader(csv_path, batch_size=64, num_workers=2, pin_memory=True, shuffle=True):
    dataset = Seq2SeqDataset(csv_path)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=collate_fn)
