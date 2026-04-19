import os
import pandas as pd
from data import vocab
from data.vocab import build_vocab_from_config, save_vocab
from utils.io import ensure_dir, save_json

def run_prepare(cfg):
    df = pd.read_csv(cfg["data"]["csv_path"])
    out_dir = cfg["data"]["output_dir"]
    ensure_dir(out_dir)

    # Mezclar para garantizar aleatoriedad reproducible
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Calcular longitudes
    max_len_plain = df["plain_text"].str.len().max()
    max_len_enc = df["caesar_encrypted"].str.len().max()
    max_len = int(max(max_len_plain, max_len_enc))
    if cfg["data"].get("use_sos_eos", False):
        max_len += 2

    # Leer proporciones desde config
    split_cfg = cfg["data"]["split"]
    train_ratio = split_cfg["train"]
    eval_ratio = split_cfg["eval"]
    infer_ratio = split_cfg["infer"]

    # assert abs(train_ratio + eval_ratio + infer_ratio - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"

    # División
    n = len(df)
    n_train = int(n * train_ratio)
    n_eval = int(n * eval_ratio)
    n_infer = n - n_train - n_eval

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_eval = df.iloc[n_train:n_train + n_eval].reset_index(drop=True)
    df_infer = df.iloc[n_train + n_eval:].reset_index(drop=True)

    # Vocabulario solo con train + eval
    vocab = build_vocab_from_config(cfg, df_train, df_eval)
    save_vocab(vocab, os.path.join(out_dir, "vocab.json"))

    # Guardar metadatos
    meta = {
        "max_len": max_len,
        "train_size": len(df_train),
        "val_size": len(df_eval),
        "infer_size": len(df_infer),
        "use_sos_eos": cfg["data"].get("use_sos_eos", False),
    }
    save_json(meta, os.path.join(out_dir, "meta.json"))

    # Guardar splits
    df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_eval.to_csv(os.path.join(out_dir, "eval.csv"), index=False)
    df_infer.to_csv(os.path.join(out_dir, "infer.csv"), index=False)

    print(f"[prepare] max_len={max_len}, train={len(df_train)}, eval={len(df_eval)}, infer={len(df_infer)}")