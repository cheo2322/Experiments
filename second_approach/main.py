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
    # ⚙️ Configuración fija
    CONFIG_PATH = "second_approach/config.yaml"
    CKPT_PATH = "artifacts/best.pt"
    TEXT = "krod"  # texto cifrado para inferencia puntual

    # 🧠 Inicialización
    cfg = load_config(CONFIG_PATH)
    set_seed(42)
    ensure_dir(cfg["data"]["output_dir"])
    device = resolve_device(cfg["train"]["device"])

    # 🧩 Paso 1: Preparación de datos
    print("📦 Preparando datos...")
    run_prepare(cfg)

    # 🚀 Paso 2: Entrenamiento
    print("🎯 Entrenando modelo...")
    run_train(cfg, device=device, resume_path=None)

    # 🧪 Paso 3: Evaluación
    print("🧪 Evaluando modelo...")
    run_evaluate(cfg, device=device, ckpt_path=CKPT_PATH)

    # 🔍 Paso 4: Inferencia puntual
    print(f"🔍 Inferencia sobre '{TEXT}'...")
    run_infer(cfg, device=device, ckpt_path=CKPT_PATH, encrypted_text=TEXT)

if __name__ == "__main__":
    main()