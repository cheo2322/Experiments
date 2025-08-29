import os, json, torch
from torch.utils.data import DataLoader
from data.dataset import CaesarDataset
from utils.io import load_json

def run_evaluate(cfg, device, ckpt_path):
    out_dir = cfg["data"]["output_dir"]
    vocab = json.load(open(os.path.join(out_dir, "vocab.json"), "r", encoding="utf-8"))
    meta = load_json(os.path.join(out_dir, "meta.json"))

    ds = CaesarDataset(os.path.join(out_dir, "val.csv"), vocab, meta["max_len"], use_sos_eos=meta["use_sos_eos"])
    loader = DataLoader(ds, batch_size=cfg["loader"]["batch_size"], shuffle=False)

    from models.seq2seq import Encoder, Decoder, Seq2Seq
    vocab_size = len(vocab["itos"])
    model = Seq2Seq(Encoder(vocab_size, cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"]),
                    Decoder(vocab_size, cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"]), device).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    char_acc, seq_acc, n_chars, n_seq = 0, 0, 0, 0
    with torch.no_grad():
        for src, trg in loader:
            src = src.to(device)
            out = model(src, trg=src, teacher_forcing_ratio=0.0)  # trg dummy; no teacher forcing
            pred = out.argmax(-1).cpu()
            trg = trg  # CPU no necesario si ya está

            # exactitud por carácter y por secuencia
            eq = (pred == trg).numpy()
            pad_id = vocab["stoi"]["<pad>"]
            mask = (trg != pad_id).numpy()
            char_acc += (eq & mask).sum()
            n_chars += mask.sum()
            seq_acc += ((eq | ~mask).all(axis=1)).sum()
            n_seq += eq.shape[0]

    print(f"[eval] char_acc={char_acc/n_chars:.4f} seq_acc={seq_acc/n_seq:.4f}")