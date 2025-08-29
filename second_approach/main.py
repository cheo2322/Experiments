import argparse
import yaml
import torch

from utils.seed import set_seed
from utils.io import ensure_dir

from pipeline.prepare import run_prepare
from pipeline.train import run_train
from pipeline.evaluate import run_evaluate
from pipeline.infer import run_infer

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_device(cfg_device):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)

def main():
    parser = argparse.ArgumentParser(description="Caesar seq2seq pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo de configuración")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    sp_prepare = subparsers.add_parser("prepare", help="Construir vocab, splits, metadatos")
    sp_prepare.add_argument("--csv", type=str, help="CSV de entrada (override de config)")

    sp_train = subparsers.add_parser("train", help="Entrenamiento")
    sp_train.add_argument("--resume", type=str, default=None, help="Checkpoint para reanudar")

    sp_eval = subparsers.add_parser("eval", help="Evaluación")
    sp_eval.add_argument("--ckpt", type=str, required=True, help="Checkpoint a evaluar")

    sp_infer = subparsers.add_parser("infer", help="Inferencia puntual")
    sp_infer.add_argument("--ckpt", type=str, required=True)
    sp_infer.add_argument("--text", type=str, required=True, help="Texto encriptado a decodificar")

    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(42)

    out_dir = cfg["data"]["output_dir"]
    ensure_dir(out_dir)

    if args.cmd == "prepare":
        if args.csv:
            cfg["data"]["csv_path"] = args.csv
        run_prepare(cfg)

    elif args.cmd == "train":
        device = resolve_device(cfg["train"]["device"])
        run_train(cfg, device=device, resume_path=args.resume)

    elif args.cmd == "eval":
        device = resolve_device(cfg["train"]["device"])
        run_evaluate(cfg, device=device, ckpt_path=args.ckpt)

    elif args.cmd == "infer":
        device = resolve_device(cfg["train"]["device"])
        run_infer(cfg, device=device, ckpt_path=args.ckpt, encrypted_text=args.text)

if __name__ == "__main__":
    main()