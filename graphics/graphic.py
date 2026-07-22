import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_accs, val_accs,
                 greedy_accs=None, exact_accs=None):
    """
    train_losses: lista con loss promedio por época en entrenamiento
    val_losses: lista con loss promedio por época en validación
    train_accs: lista con accuracy en entrenamiento por época
    val_accs: lista con accuracy en validación por época
    greedy_accs: lista con accuracy por token en greedy decoding (opcional)
    exact_accs: lista con exact match en greedy decoding (opcional)
    """
    epochs = list(range(1, len(train_losses)+1))

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # Losses
    axes[0].plot(epochs, train_losses, label="Train Loss", color="blue")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="orange")
    axes[0].set_title("Epoch loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracies
    axes[1].plot(epochs, train_accs, label="Train Accuracy", color="green")
    axes[1].plot(epochs, val_accs, label="Val Accuracy (Teacher Forcing)", color="red")
    if greedy_accs is not None:
        axes[1].plot(epochs, greedy_accs, label="Greedy Accuracy", color="purple", linestyle="--")
    if exact_accs is not None:
        axes[1].plot(epochs, exact_accs, label="Exact Match", color="brown", linestyle=":")
    axes[1].set_title("Epoch accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle("Training and validation metrics", fontsize=16)
    plt.tight_layout()
    plt.savefig("t1_third_approach/artifacts/metrics.png")
    plt.close(fig)
