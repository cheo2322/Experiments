import torch


def compute_accuracy(predictions, targets, pad_idx=0):
    mask = targets != pad_idx
    correct = (predictions.argmax(-1) == targets) & mask
    return correct.sum().item(), mask.sum().item()


def eval_model(eval_loader, model, criterion, device, pad_idx=0):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for plain, encrypted, plain_lengths, _ in eval_loader:
            plain, encrypted = plain.to(device), encrypted.to(device)

            # forward
            output = model(plain, plain_lengths, encrypted[:, :-1])

            # loss
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                encrypted[:, 1:].reshape(-1)
            )

            # número de tokens válidos (sin pad)
            num_tokens = (encrypted[:, 1:] != pad_idx).sum().item()

            # acumular pérdida ponderada por tokens
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # accuracy con la función
            correct, _ = compute_accuracy(output, encrypted[:, 1:], pad_idx)
            total_correct += correct
            # ojo: tokens aquí y num_tokens arriba deberían coincidir
            # pero usamos ambos para consistencia
            # total_tokens ya se acumula arriba

    # pérdida promedio por token
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    # accuracy promedio por token
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc
