def build_vocab_from_config(cfg, df_train, df_val):
    fixed = list(cfg["vocab"]["fixed_chars"])
    add = cfg["vocab"]["add_tokens"]
    itos = fixed + [t for t in add if t not in fixed]
    stoi = {c: i for i, c in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}

def save_vocab(vocab, path):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)