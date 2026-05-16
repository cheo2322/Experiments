import torch


def compute_accuracy(predictions, targets, pad_idx=0):
    mask = targets != pad_idx
    correct = (predictions.argmax(-1) == targets) & mask
    return correct.sum().item() / mask.sum().item()

def eval_model(eval_loader, model, criterion, device, pad_idx=0):
    model.eval()
    total_loss, total_acc = 0, 0
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
            total_loss += loss.item()

            # accuracy
            total_acc += compute_accuracy(output, encrypted[:, 1:], pad_idx)

    avg_loss = total_loss / len(eval_loader)
    avg_acc = total_acc / len(eval_loader)
    return avg_loss, avg_acc