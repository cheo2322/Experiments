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

def main():
    # Leer configuración
    config = load_config("t1_third_approach/config.yaml")
    device = resolve_device(config["train"]["device"])
    
    print("Device:", device)

    # Extraer parámetros de la sección 'data'
    csv_path = config["data"]["csv_path"]
    min_len = config["data"].get("min_len", 5)
    max_len = config["data"].get("max_len", 12)
    n = config["data"].get("n", 10)
    key_str = config["data"].get("key", "secretkey")

    # Generate dataset (uncomment if you want to generate a new one, but be careful as it will overwrite the existing one)
    rc4.generate_csv(csv_path, n, min_len, max_len, key_str)

    # Dataset completo
    dataset = Seq2SeqDataset(csv_path, add_sos_eos=True)

    # Calcular tamaños según split del yaml
    n_total = len(dataset)
    n_train = int(config["data"]["split"]["train"] * n_total)
    n_eval = int(config["data"]["split"]["eval"] * n_total)
    n_infer = n_total - n_train - n_eval

    # Dividir dataset
    train_set, eval_set, infer_set = random_split(dataset, [n_train, n_eval, n_infer])

    # Crear DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["train"]["num_workers"])

    eval_loader = DataLoader(
        eval_set,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["train"]["num_workers"])

    infer_loader = DataLoader(
        infer_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["train"]["num_workers"])
    
    # Training
    vocab_size = config["train"]["vocab_size"]
    train_model(
        [train_loader, eval_loader],
        vocab_size=vocab_size,
        emb_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        lr=config["train"]["lr"],
        epochs=config["train"]["epochs"],
        grad_clip=config["train"]["grad_clip"],
        device=device,
        output_dir=config["data"]["output_dir"],
        weight_decay=config["train"]["weight_decay"],
        print_every=config["train"]["print_every"],
        loss_threshold=config["validation"]["loss_threshold"],
        acc_threshold=config["validation"]["acc_threshold"]
    )
    
    # Inference
    inference = infer_model(
        infer_loader,
        dataset,
        device,
        vocab_size=config["train"]["vocab_size"],
        embidding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        output_dir=config["data"]["output_dir"]
    )
    
    greedy_acc = inference["greedy_acc"]
    exact_acc = inference["exact_acc"]
    
    import matplotlib.pyplot as plt

    def plot_greedy_metrics(greedy_acc, exact_acc, title="Greedy inference metrics"):
        fig, ax = plt.subplots(figsize=(6,5))

        metrics = ["Greedy Accuracy", "Exact Match"]
        values = [greedy_acc, exact_acc]

        ax.bar(metrics, values, color=["purple", "brown"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

        plt.tight_layout()
        plt.savefig("t1_third_approach/artifacts/greedy_metrics.png")
        plt.close(fig)

    plot_greedy_metrics(greedy_acc, exact_acc)


if __name__ == "__main__":
    main()
