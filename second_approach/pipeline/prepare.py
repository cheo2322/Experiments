import os
import pandas as pd
from data.vocab import build_vocab_from_config, save_vocab
from utils.io import ensure_dir, save_json

def run_prepare(cfg):
    df = pd.read_csv(cfg["data"]["csv_path"])
    out_dir = cfg["data"]["output_dir"]
    ensure_dir(out_dir)

    # Calcular longitudes
    max_len_plain = df["plain_text"].str.len().max()
    max_len_enc = df["caesar_encrypted"].str.len().max()
    max_len = int(max(max_len_plain, max_len_enc))
    if cfg["data"].get("use_sos_eos", False):
        max_len += 2

    # Split
    split = cfg["data"]["train_split"]
    n_train = int(len(df) * split)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train:].reset_index(drop=True)

    # Vocab
    vocab = build_vocab_from_config(cfg, df_train, df_val)
    save_vocab(vocab, os.path.join(out_dir, "vocab.json"))

    # Guardar metadatos
    meta = {
        "max_len": max_len,
        "train_size": len(df_train),
        "val_size": len(df_val),
        "use_sos_eos": cfg["data"].get("use_sos_eos", False),
    }
    save_json(meta, os.path.join(out_dir, "meta.json"))

    # Persistir splits
    df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(out_dir, "val.csv"), index=False)

    print(f"[prepare] max_len={max_len}, train={len(df_train)}, val={len(df_val)}")