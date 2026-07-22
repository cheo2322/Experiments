import torch


def compute_accuracy(predictions, targets, pad_idx=0):
    mask = targets != pad_idx
    correct = (predictions.argmax(-1) == targets) & mask
    return correct.sum().item(), mask.sum().item()

def greedy_decode(model, encrypted, max_len, sos_idx, eos_idx, device):
    """
    Greedy decoding para batch completo.
    Genera secuencias en paralelo hasta que todas lleguen a <eos> o max_len.
    """
    model.eval()
    with torch.no_grad():
        # Codificar el batch
        memory, src_key_padding_mask = model.encoder(encrypted)
        batch_size = encrypted.size(0)

        # Inicializar con <sos> para cada ejemplo
        input_token = torch.full((batch_size, 1), sos_idx, device=device)

        # Lista de secuencias decodificadas (una por ejemplo)
        decoded = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            output = model.decoder(input_token, memory, memory_key_padding_mask=src_key_padding_mask)
            next_token = output.argmax(-1)[:, -1]  # (batch_size,)

            # Actualizar secuencias
            for i in range(batch_size):
                if not finished[i]:
                    if next_token[i].item() == eos_idx:
                        finished[i] = True
                    else:
                        decoded[i].append(next_token[i].item())

            # Preparar input para el siguiente paso
            input_token = next_token.unsqueeze(1)

            # Si todos terminaron, salir
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
            for pred_seq, target_seq in zip(pred_batch, plain.cpu().tolist()):
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
