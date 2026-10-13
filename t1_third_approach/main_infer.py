import yaml
import torch
from torch.utils.data import random_split, DataLoader

from t1_third_approach.data import rc4
from dataloader.dataloader_seq_2_seq import Seq2SeqDataset, collate_fn
from t1_third_approach.pipeline.train_model import train_model
from t1_third_approach.pipeline.infer_model import infer_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_device(cfg_device):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)

# Claves a probar. La primera es la de entrenamiento y actúa como control:
# si esta fila no da ~100%, el problema está en el script y no en el modelo.
TEST_KEYS = ["secretkey", "secretkez", "anotherkey", "k"]
N_SAMPLES = 1000


def main():
    # Leer configuración
    config = load_config("t1_third_approach/config.yaml")
    device = resolve_device(config["train"]["device"])
    
    print("Device:", device)

    # Extraer parámetros de la sección 'data'
    # csv_path = config["data"]["csv_path"]   # <- no se usa: cada clave escribe su propio CSV
    min_len = config["data"].get("min_len", 5)
    max_len = config["data"].get("max_len", 12)
    # n = config["data"].get("n", 10)         # <- se reemplaza por N_SAMPLES
    # key_str = config["data"].get("key", "secretkey")   # <- se reemplaza por TEST_KEYS

    results = []

    for key_str in TEST_KEYS:
        # CSV aparte por clave para NO sobreescribir el dataset original
        csv_path = f"t1_third_approach/artifacts/crosskey_{key_str}.csv"
        rc4.generate_csv(csv_path, N_SAMPLES, min_len, max_len, key_str)

        # Dataset completo: aquí no se entrena, así que todo va a inferencia
        dataset = Seq2SeqDataset(csv_path)

        # --- Split innecesario: no hay entrenamiento en este script ---------
        # n_total = len(dataset)
        # n_train = int(config["data"]["split"]["train"] * n_total)
        # n_eval = int(config["data"]["split"]["eval"] * n_total)
        # n_infer = n_total - n_train - n_eval
        # train_set, eval_set, infer_set = random_split(dataset, [n_train, n_eval, n_infer])

        # --- DataLoaders de train/eval innecesarios -------------------------
        # train_loader = DataLoader(...)
        # eval_loader = DataLoader(...)

        infer_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config["train"]["num_workers"])

        # --- Training: NO se ejecuta. Se reusan los pesos ya entrenados -----
        # train_model(...)

        # Inference
        inference = infer_model(
            infer_loader,
            dataset,
            device,
            vocab_size=config["train"]["vocab_size"],
            embidding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            output_dir=config["data"]["output_dir"],
            max_len=config["model"].get("max_len", 64)
        )

        results.append((key_str, inference["token_acc"], inference["exact_acc"]))
        print(f"  {key_str}: token={inference['token_acc']:.4f} exact={inference['exact_acc']:.4f}")

    # Tabla final
    print()
    print(f"{'key':<15}{'token acc':>12}{'exact acc':>12}")
    print("-" * 39)
    for key_str, token_acc, exact_acc in results:
        label = key_str + (" *" if key_str == TEST_KEYS[0] else "")
        print(f"{label:<15}{token_acc:>11.2%}{exact_acc:>12.2%}")
    print("\n* clave de entrenamiento (control)")

    # --- Gráfico de una sola corrida: no aplica a una tabla de varias claves ---
    # import matplotlib.pyplot as plt
    # def plot_inference_metrics(token_acc, exact_acc, title="Inference metrics"):
    #     ...
    # plot_inference_metrics(token_acc, exact_acc)


if __name__ == "__main__":
    main()