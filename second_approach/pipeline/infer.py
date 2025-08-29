import os, json, torch
from utils.io import load_json

def greedy_decode(model, src, max_len):
    model.eval()
    with torch.no_grad():
        # bootstrap con src como guía; si usas <sos>/<eos>, cambia aquí
        out = model(src, trg=src, teacher_forcing_ratio=0.0)
        return out.argmax(-1)

def run_infer(cfg, device, ckpt_path, encrypted_text):
    out_dir = cfg["data"]["output_dir"]
    vocab = json.load(open(os.path.join(out_dir, "vocab.json"), "r", encoding="utf-8"))
    meta = load_json(os.path.join(out_dir, "meta.json"))

    from data.dataset import encode_text
    from models.seq2seq import Encoder, Decoder, Seq2Seq

    vocab_size = len(vocab["itos"])
    model = Seq2Seq(Encoder(vocab_size, cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"]),
                    Decoder(vocab_size, cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"]),
                    device).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    ids = encode_text(encrypted_text, vocab, meta["max_len"], use_sos_eos=meta["use_sos_eos"]).unsqueeze(0).to(device)
    pred = greedy_decode(model, ids, meta["max_len"]).squeeze(0).cpu().tolist()

    itos = vocab["itos"]
    pad_id = vocab["stoi"]["<pad>"]
    decoded = ''.join(itos[i] for i in pred if i != pad_id)
    print(f"[infer] enc='{encrypted_text}' -> dec='{decoded}'")