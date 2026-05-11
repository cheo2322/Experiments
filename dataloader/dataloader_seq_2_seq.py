import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Seq2SeqDataset(Dataset):
    def __init__(self, csv_path, add_sos_eos=True):
        # Cargar CSV
        df = pd.read_csv(csv_path)
        self.plain_texts = df["plain"].tolist()
        self.encrypted_texts = df["encrypted"].tolist()
        self.add_sos_eos = add_sos_eos

        # Vocab de 256 ASCII + tokens especiales
        base_vocab = [chr(i) for i in range(256)]
        self.vocab = base_vocab + ["<pad>", "<sos>", "<eos>"]

        self.char2idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}

    def __len__(self):
        return len(self.plain_texts)

    def encode(self, text):
        seq = [self.char2idx[ch] for ch in text if ch in self.char2idx]
        if self.add_sos_eos:
            seq = [self.char2idx["<sos>"]] + seq + [self.char2idx["<eos>"]]
        return seq

    def __getitem__(self, idx):
        plain_seq = self.encode(self.plain_texts[idx])
        encrypted_seq = self.encode(self.encrypted_texts[idx])
        return torch.tensor(plain_seq, dtype=torch.long), torch.tensor(encrypted_seq, dtype=torch.long)


def collate_fn(batch):
    plain_seqs, encrypted_seqs = zip(*batch)

    plain_lengths = [len(seq) for seq in plain_seqs]
    encrypted_lengths = [len(seq) for seq in encrypted_seqs]

    plain_padded = torch.nn.utils.rnn.pad_sequence(plain_seqs, batch_first=True, padding_value=0)
    encrypted_padded = torch.nn.utils.rnn.pad_sequence(encrypted_seqs, batch_first=True, padding_value=0)

    return plain_padded, encrypted_padded, plain_lengths, encrypted_lengths


def get_dataloader(csv_path, batch_size=64, num_workers=2, pin_memory=True, add_sos_eos=True):
    dataset = Seq2SeqDataset(csv_path, add_sos_eos=add_sos_eos)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=collate_fn)
