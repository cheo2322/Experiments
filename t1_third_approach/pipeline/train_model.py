import torch
import torch.nn as nn
import torch.optim as optim

from graphics.graphic import plot_metrics
from models.seq2seq.seq2seq import PositionwiseModel


def compute_accuracy(predictions, targets, pad_idx=0):
    mask = targets != pad_idx
    correct = (predictions.argmax(-1) == targets) & mask
    return correct.sum().item(), mask.sum().item()


def eval_model(eval_loader, model, criterion, device, pad_idx=0):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    exact_matches, total_examples = 0, 0

    with torch.no_grad():
        for encrypted, plain, _, _ in eval_loader:
            encrypted, plain = encrypted.to(device), plain.to(device)
            output = model(encrypted)

            loss = criterion(output.reshape(-1, output.size(-1)), plain.reshape(-1))
            num_tokens = (plain != pad_idx).sum().item()
            total_loss += loss.item() * num_tokens

            correct, tokens = compute_accuracy(output, plain, pad_idx)
            total_correct += correct
            total_tokens += tokens

            preds = output.argmax(-1)
            for pred_row, target_row in zip(preds, plain):
                mask = target_row != pad_idx
                if torch.equal(pred_row[mask], target_row[mask]):
                    exact_matches += 1
                total_examples += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    exact_acc = exact_matches / total_examples if total_examples > 0 else 0.0

    return {"loss": avg_loss, "acc": avg_acc, "exact_acc": exact_acc}


def train_model(loaders, vocab_size, emb_dim, hidden_dim,
                lr, epochs, grad_clip, device, output_dir, weight_decay,
                print_every=1, max_len=64):

    model = PositionwiseModel(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        n_heads=4,
        n_layers=2,
        ff_dim=hidden_dim,
        max_len=max_len
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_eval_loss = float('inf')
    best_eval_acc = 0.0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies, exact_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_tokens = 0, 0, 0

        for encrypted, plain, _, _ in loaders[0]:
            encrypted, plain = encrypted.to(device), plain.to(device)
            optimizer.zero_grad()

            output = model(encrypted)
            loss = criterion(output.reshape(-1, vocab_size), plain.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            correct, tokens = compute_accuracy(output, plain, pad_idx=0)
            total_loss += loss.item() * tokens
            total_correct += correct
            total_tokens += tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        train_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

        metrics = eval_model(loaders[1], model, criterion, device)
        val_loss, val_acc, exact_acc = metrics["loss"], metrics["acc"], metrics["exact_acc"]

        scheduler.step()

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        exact_accuracies.append(exact_acc)

        if epoch % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Exact Acc: {exact_acc:.4f}")

        if val_loss < best_eval_loss and val_acc > best_eval_acc:
            best_eval_loss = val_loss
            best_eval_acc = val_acc
            ckpt_path = f"{output_dir}/best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}. Epoch {epoch+1}")
            
        if exact_acc > 0.9999:
            print(f"Early stopping at epoch {epoch+1} due to high exact match accuracy.")
            break

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, exact_accs=exact_accuracies)
