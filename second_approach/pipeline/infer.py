import os, json, torch
import pandas as pd
from data import vocab
from utils.io import load_json, save_json
from nltk.translate.bleu_score import sentence_bleu
from difflib import SequenceMatcher
from data.dataset import encode_text
from models.seq2seq import Encoder, Decoder, Seq2Seq

def greedy_decode(model, src, vocab, max_len):
    model.eval()
    sos_idx = vocab["stoi"]["<sos>"]
    pad_idx = vocab["stoi"]["<pad>"]

    batch_size = src.size(0)
    device = src.device
    vocab_size = len(vocab["itos"])

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

def run_infer(cfg, device, ckpt_path, encrypted_text=None):
    enc_dim = cfg["model"]["hidden_dim"] * 2
    dec_dim = cfg["model"]["hidden_dim"]
    
    out_dir = cfg["data"]["output_dir"]
    vocab = json.load(open(os.path.join(out_dir, "vocab.json"), "r", encoding="utf-8"))
    meta = load_json(os.path.join(out_dir, "meta.json"))

    vocab_size = len(vocab["itos"])
    model = Seq2Seq(Encoder(vocab_size, cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"]),
                    Decoder(vocab_size, cfg["model"]["emb_dim"], enc_dim, dec_dim),
                    device).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sos_idx = vocab["stoi"]["<sos>"]
    eos_idx = vocab["stoi"]["<eos>"]
    pad_id = vocab["stoi"]["<pad>"]
    itos = vocab["itos"]

    # 🔍 Inferencia puntual
    if encrypted_text:
        ids = encode_text(encrypted_text, vocab, meta["max_len"], use_sos_eos=meta["use_sos_eos"]).unsqueeze(0).to(device)
        pred = greedy_decode(model, ids, vocab, meta["max_len"]).squeeze(0).cpu().tolist()

        if eos_idx in pred:
            pred = pred[:pred.index(eos_idx)]
        if pred and pred[0] == sos_idx:
            pred = pred[1:]

        decoded = ''.join(itos[i] for i in pred if i != pad_id)
        print(f"[infer] enc='{encrypted_text}' -> dec='{decoded}'")
        return decoded

    # 📊 Inferencia masiva sobre infer.csv
    df = pd.read_csv(os.path.join(out_dir, "infer.csv"))
    encrypted_texts = df["caesar_encrypted"].tolist()
    targets = df["plain_text"].tolist()

    preds = []
    for text in encrypted_texts:
        ids = encode_text(text, vocab, meta["max_len"], use_sos_eos=meta["use_sos_eos"]).unsqueeze(0).to(device)
        pred = greedy_decode(model, ids, vocab, meta["max_len"]).squeeze(0).cpu().tolist()

        if eos_idx in pred:
            pred = pred[:pred.index(eos_idx)]
        if pred and pred[0] == sos_idx:
            pred = pred[1:]

        decoded = ''.join(itos[i] for i in pred if i != pad_id)
        preds.append(decoded)

    # 🧠 Métricas
    def levenshtein(a, b):
        return int((1 - SequenceMatcher(None, a, b).ratio()) * max(len(a), len(b)))

    total_chars = sum(len(t) for t in targets)
    correct_chars = sum(sum(p == t for p, t in zip(pred, tgt)) for pred, tgt in zip(preds, targets))
    exact_matches = sum(int(p == t) for p, t in zip(preds, targets))
    bleu_scores = [sentence_bleu([list(t)], list(p)) for p, t in zip(preds, targets)]
    levenshtein_total = sum(levenshtein(p, t) for p, t in zip(preds, targets))

    metrics = {
        "char_accuracy": round(correct_chars / total_chars, 4),
        "exact_match": round(exact_matches / len(preds), 4),
        "avg_bleu": round(sum(bleu_scores) / len(bleu_scores), 4),
        "avg_levenshtein": round(levenshtein_total / len(preds), 4)
    }

    print("📊 Métricas de inferencia:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    save_json(metrics, os.path.join(out_dir, "infer_metrics.json"))