import os
import json
from xml.parsers.expat import model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import CaesarDataset
from models.seq2seq import Encoder, Decoder, Seq2Seq
from utils.io import load_json, save_ckpt

def audit_batch(pred, trg, vocab, max_examples=3):
    for i in range(min(max_examples, pred.shape[0])):
        pred_tokens = [vocab["itos"][idx] for idx in pred[i].tolist()]
        target_tokens = [vocab["itos"][idx] for idx in trg[i].tolist()]
        print(f"[audit] pred[{i}]:  {pred_tokens}")
        print(f"[audit] trg [{i}]:  {target_tokens}")
        
def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)

def run_train(cfg, device, resume_path=None):
    enc_dim = cfg["model"]["hidden_dim"] * 2
    dec_dim = cfg["model"]["hidden_dim"]

    out_dir = cfg["data"]["output_dir"]
    vocab = json.load(open(os.path.join(out_dir, "vocab.json"), "r", encoding="utf-8"))
    meta = load_json(os.path.join(out_dir, "meta.json"))

    train_ds = CaesarDataset(os.path.join(out_dir, "train.csv"), vocab, meta["max_len"], use_sos_eos=meta["use_sos_eos"])
    val_ds = CaesarDataset(os.path.join(out_dir, "val.csv"), vocab, meta["max_len"], use_sos_eos=meta["use_sos_eos"])

    loader_tr = DataLoader(train_ds, batch_size=cfg["loader"]["batch_size"], shuffle=True, num_workers=cfg["loader"]["num_workers"], pin_memory=cfg["loader"]["pin_memory"])
    loader_va = DataLoader(val_ds, batch_size=cfg["loader"]["batch_size"], shuffle=False, num_workers=cfg["loader"]["num_workers"], pin_memory=cfg["loader"]["pin_memory"])

    vocab_size = len(vocab["itos"])
    enc = Encoder(vocab_size, cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"])
    dec = Decoder(vocab_size, cfg["model"]["emb_dim"], enc_dim, dec_dim)
    model = Seq2Seq(enc, dec, device).to(device)
    initialize_weights(model)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab["stoi"]["<pad>"])
    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    start_epoch = 1
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[train] Resumido desde {resume_path}, epoch={start_epoch}")

    best_val = float("inf")

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        total = 0.0
        for src, trg in loader_tr:
            src, trg = src.to(device), trg.to(device)
            
            # for i in range(cfg["loader"]["batch_size"]):
            #     print(f"[audit] target[{i}]: {[vocab['itos'][idx] for idx in trg[i].tolist()]}")
                
            optim.zero_grad()
            forced_token_id = vocab["stoi"]["<sos>"]
            out = model(src, trg, teacher_forcing_ratio=cfg["train"]["teacher_forcing"], forced_token_id=forced_token_id)
            
            
            # Auditoría del primer token predicho en el paso t=1
            # first_logits = out[:, 1, :]  # [B, vocab_size]
            # first_pred = first_logits.argmax(dim=1)  # [B]

            # from collections import Counter
            # counts = Counter(first_pred.cpu().tolist())
            # tokens = [vocab["itos"][idx] for idx in first_pred.cpu().tolist()]
            # print(f"[epoch {epoch}] distribución primer token:", dict(counts))
            # print(f"[epoch {epoch}] primer tokens predichos:", tokens[:10])  # muestra los primeros 10

            # shift para saltar el primer paso
            logits = out[:, 1:].reshape(-1, vocab_size)
            targets = trg[:, 1:].reshape(-1)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optim.step()
            total += loss.item()

        val_loss = evaluate_loss(model, loader_va, criterion, device, vocab_size, vocab)
        print(f"[epoch {epoch}] train_loss={total/len(loader_tr):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(os.path.join(out_dir, f"best.pt"), model, optim, epoch)
        if (epoch % cfg["train"]["ckpt_every"]) == 0:
            save_ckpt(os.path.join(out_dir, f"epoch_{epoch}.pt"), model, optim, epoch)

def evaluate_loss(model, loader, criterion, device, vocab_size, vocab):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            out = model(src, trg, teacher_forcing_ratio=0.0)
            
            pred = out.argmax(dim=-1)
            audit_batch(pred, trg, vocab)
            
            logits = out[:, 1:].reshape(-1, vocab_size)
            targets = trg[:, 1:].reshape(-1)
            loss = criterion(logits, targets)
            tot += loss.item()
    return tot / len(loader)