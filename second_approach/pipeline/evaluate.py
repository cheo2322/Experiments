from data.dataset import CaesarDataset
from utils.io import load_json
from models.seq2seq import Encoder, Decoder, Seq2Seq
from data.dataset import encode_text
from difflib import SequenceMatcher
import torch
import os
import json
from torch.utils.data import DataLoader


def greedy_decode(model, src, vocab, max_len):
    model.eval()
    sos_idx = vocab["stoi"]["<sos>"]
    pad_idx = vocab["stoi"]["<pad>"]

    batch_size = src.size(0)
    device = src.device

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        outputs = []

        for _ in range(max_len):
            output, hidden = model.decoder(input, hidden, encoder_outputs)
            top1 = output.argmax(1).unsqueeze(1)
            outputs.append(top1)
            input = top1

        pred = torch.cat(outputs, dim=1)
        return pred

def run_evaluate(cfg, device, ckpt_path):
    enc_dim = cfg["model"]["hidden_dim"] * 2
    dec_dim = cfg["model"]["hidden_dim"]

    out_dir = cfg["data"]["output_dir"]
    vocab = json.load(open(os.path.join(out_dir, "vocab.json"), "r", encoding="utf-8"))
    meta = load_json(os.path.join(out_dir, "meta.json"))

    ds = CaesarDataset(os.path.join(out_dir, "eval.csv"), vocab, meta["max_len"], use_sos_eos=meta["use_sos_eos"])
    loader = DataLoader(ds, batch_size=cfg["loader"]["batch_size"], shuffle=False)

    vocab_size = len(vocab["itos"])
    model = Seq2Seq(Encoder(vocab_size, cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"]),
                    Decoder(vocab_size, cfg["model"]["emb_dim"], enc_dim, dec_dim), device).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sos_idx = vocab["stoi"]["<sos>"]
    eos_idx = vocab["stoi"]["<eos>"]
    pad_id = vocab["stoi"]["<pad>"]
    itos = vocab["itos"]

    char_acc, seq_acc, n_chars, n_seq = 0, 0, 0, 0

    with torch.no_grad():
        for src, trg in loader:
            src = src.to(device)
            pred_ids = greedy_decode(model, src, vocab, meta["max_len"]).cpu()

            for pred_seq, target_seq in zip(pred_ids, trg):
                # Truncar en <eos>
                pred = pred_seq.tolist()
                if eos_idx in pred:
                    pred = pred[:pred.index(eos_idx)]
                if pred and pred[0] == sos_idx:
                    pred = pred[1:]
                pred_text = ''.join(itos[i] for i in pred if i != pad_id)

                target = target_seq.tolist()
                if eos_idx in target:
                    target = target[:target.index(eos_idx)]
                if target and target[0] == sos_idx:
                    target = target[1:]
                target_text = ''.join(itos[i] for i in target if i != pad_id)

                # Métricas
                n_seq += 1
                n_chars += len(target_text)
                char_acc += sum(p == t for p, t in zip(pred_text, target_text))
                seq_acc += int(pred_text == target_text)

    print(f"[eval] char_acc={char_acc/n_chars:.4f} seq_acc={seq_acc/n_seq:.4f}")