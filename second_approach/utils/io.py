import os, json, torch

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_json(obj, path): 
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
def load_json(path): 
    with open(path, "r", encoding="utf-8") as f: return json.load(f)
def save_ckpt(path, model, optim, epoch):
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch}, path)
