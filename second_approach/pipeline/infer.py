import os, json, torch
from data import vocab
from utils.io import load_json

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
            top1 = output.argmax(1).unsqueeze(1)  # [B, 1]
            outputs.append(top1)
            input = top1

        pred = torch.cat(outputs, dim=1)  # [B, T]
        return pred

def run_infer(cfg, device, ckpt_path, encrypted_text):
    enc_dim = cfg["model"]["hidden_dim"] * 2
    dec_dim = cfg["model"]["hidden_dim"]
    
    out_dir = cfg["data"]["output_dir"]
    vocab = json.load(open(os.path.join(out_dir, "vocab.json"), "r", encoding="utf-8"))
    meta = load_json(os.path.join(out_dir, "meta.json"))

    from data.dataset import encode_text
    from models.seq2seq import Encoder, Decoder, Seq2Seq

    vocab_size = len(vocab["itos"])
    model = Seq2Seq(Encoder(vocab_size, cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"]),
                    Decoder(vocab_size, cfg["model"]["emb_dim"], enc_dim, dec_dim),
                    device).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    ids = encode_text(encrypted_text, vocab, meta["max_len"], use_sos_eos=meta["use_sos_eos"]).unsqueeze(0).to(device)
    pred = greedy_decode(model, ids, vocab, meta["max_len"]).squeeze(0).cpu().tolist()
    
    sos_idx = vocab["stoi"]["<sos>"]
    eos_idx = vocab["stoi"]["<eos>"]

    # Cortar en <eos> si aparece
    if eos_idx in pred:
        pred = pred[:pred.index(eos_idx)]

    # Eliminar <sos> si aparece al inicio (por error del modelo)
    if pred and pred[0] == sos_idx:
        pred = pred[1:]

    itos = vocab["itos"]
    pad_id = vocab["stoi"]["<pad>"]
    decoded = ''.join(itos[i] for i in pred if i != pad_id)
    print(f"[infer] enc='{encrypted_text}' -> dec='{decoded}'")