import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import base64

class Seq2SeqDataset(Dataset):
    def __init__(self, csv_path, add_sos_eos=True):
        # Cargar CSV
        df = pd.read_csv(csv_path)
        self.plain_texts = df["plain"].tolist()
        self.encrypted_texts = df["encrypted"].tolist()
        self.add_sos_eos = add_sos_eos

        # Vocab de 256 ASCII + tokens especiales
        special_tokens = ["<pad>", "<sos>", "<eos>"]
        base_vocab = [chr(i) for i in range(256)]
        self.vocab = special_tokens + base_vocab

        self.char2idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}

        self.pad_idx = self.char2idx["<pad>"]  # ahora sí es 0

    def __len__(self):
        return len(self.plain_texts)

    def encode(self, text):
        seq = [self.char2idx[ch] for ch in text if ch in self.char2idx]
        if self.add_sos_eos:
            seq = [self.char2idx["<sos>"]] + seq + [self.char2idx["<eos>"]]
        return seq

    def __getitem__(self, idx):
        plain_seq = self.encode(self.plain_texts[idx])

        # Decodificar Base64 antes de tokenizar
        encrypted_b64 = self.encrypted_texts[idx]
        encrypted_raw = base64.b64decode(encrypted_b64).decode("latin-1")

        encrypted_seq = self.encode(encrypted_raw)
        return torch.tensor(encrypted_seq, dtype=torch.long), torch.tensor(plain_seq, dtype=torch.long)


def collate_fn(batch):
    encrypted_seqs, plain_seqs = zip(*batch)

    encrypted_lengths = [len(seq) for seq in encrypted_seqs]
    plain_lengths = [len(seq) for seq in plain_seqs]

    encrypted_padded = torch.nn.utils.rnn.pad_sequence(encrypted_seqs, batch_first=True, padding_value=0)
    plain_padded = torch.nn.utils.rnn.pad_sequence(plain_seqs, batch_first=True, padding_value=0)

    return encrypted_padded, plain_padded, encrypted_lengths, plain_lengths


def get_dataloader(csv_path, batch_size=64, num_workers=2, pin_memory=True, add_sos_eos=True):
    dataset = Seq2SeqDataset(csv_path, add_sos_eos=add_sos_eos)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=collate_fn)
