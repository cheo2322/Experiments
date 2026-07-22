import torch


def compute_accuracy(predictions, targets, pad_idx=0):
    mask = targets != pad_idx
    correct = (predictions.argmax(-1) == targets) & mask
    return correct.sum().item(), mask.sum().item()

def greedy_decode(model, encrypted, max_len, sos_idx, eos_idx, device):
    model.eval()
    with torch.no_grad():
        memory, src_key_padding_mask = model.encoder(encrypted)
        batch_size = encrypted.size(0)

        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        decoded = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            output = model.decoder(generated, memory, memory_key_padding_mask=src_key_padding_mask)
            next_token = output.argmax(-1)[:, -1]  # último paso de la secuencia completa

            for i in range(batch_size):
                if not finished[i]:
                    if next_token[i].item() == eos_idx:
                        finished[i] = True
                    else:
                        decoded[i].append(next_token[i].item())

            # Concatenar, no reemplazar
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

    return decoded


def eval_model(eval_loader, model, criterion, device, pad_idx=0, sos_idx=1, eos_idx=2):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    greedy_correct, greedy_tokens, exact_matches = 0, 0, 0

    with torch.no_grad():
        for encrypted, plain, encrypted_lengths, _ in eval_loader:
            encrypted, plain = encrypted.to(device), plain.to(device)

            # Teacher forcing forward
            output = model(encrypted, plain[:, :-1])
            
            # Training loss
            loss = criterion(output.reshape(-1, output.size(-1)), plain[:, 1:].reshape(-1))

            # Tokens válidos del target
            num_tokens = (plain[:, 1:] != pad_idx).sum().item()
            total_loss += loss.item() * num_tokens

            # Teacher forcing accuracy
            correct, tokens = compute_accuracy(output, plain[:, 1:], pad_idx)
            total_correct += correct
            total_tokens += tokens

            # Greedy decoding para todo el batch
            pred_batch = greedy_decode(model, encrypted, max_len=plain.size(1), sos_idx=sos_idx, eos_idx=eos_idx, device=device)

            # Comparar cada ejemplo del batch
            for pred_seq, target_row in zip(pred_batch, plain.cpu().tolist()):
                # Quitar <sos> inicial, y recortar en el primer <eos> o <pad>
                target_seq = target_row[1:]
                if eos_idx in target_seq:
                    target_seq = target_seq[:target_seq.index(eos_idx)]
                elif pad_idx in target_seq:
                    target_seq = target_seq[:target_seq.index(pad_idx)]

                min_len = min(len(pred_seq), len(target_seq))
                greedy_correct += sum(p == t for p, t in zip(pred_seq[:min_len], target_seq[:min_len]))
                greedy_tokens += len(target_seq)

                if pred_seq == target_seq:
                    exact_matches += 1

    # Métricas finales
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    greedy_acc = greedy_correct / greedy_tokens if greedy_tokens > 0 else 0.0
    exact_acc = exact_matches / len(eval_loader.dataset) if len(eval_loader.dataset) > 0 else 0.0

    return {
        "loss": avg_loss,
        "teacher_acc": avg_acc,
        "greedy_acc": greedy_acc,
        "exact_acc": exact_acc
    }
