import torch
from models.seq2seq.seq2seq import Seq2Seq, TransformerEncoder, TransformerDecoder

def decode_tokens(seq, vocab):
    chars = []
    for token in seq:  # token debe ser int
        ch = vocab[token]
        if ch in ("<sos>", "<pad>"):
            continue
        if ch == "<eos>":
            break
        chars.append(ch)
    return "".join(chars)

def greedy_decode(model, encrypted, encrypted_lengths, max_len, sos_idx, eos_idx, device):
    # batch size = 1 en inferencia
    hidden = model.encoder(encrypted, encrypted_lengths)
    input_token = torch.tensor([[sos_idx]], device=device)
    decoded = []
    for _ in range(max_len):
        output, hidden = model.decoder(input_token, hidden)
        next_token = output.argmax(-1)[:, -1]
        if next_token.item() == eos_idx:
            break
        decoded.append(next_token.item())
        input_token = next_token.unsqueeze(0)
    return decoded

def infer_model(infer_loader, dataset, device, vocab_size, embidding_dim, hidden_dim, output_dir,
                sos_idx=1, eos_idx=2):

    # Rebuild model
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        emb_dim=embidding_dim,
        n_heads=4,
        n_layers=2,
        ff_dim=256
    )
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        emb_dim=embidding_dim,
        n_heads=4,
        n_layers=2,
        ff_dim=256
    )
    model = Seq2Seq(encoder, decoder).to(device)

    # Load checkpoint
    ckpt_path = f"{output_dir}/best_model.pt"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    predictions = []
    greedy_correct, greedy_tokens, exact_matches = 0, 0, 0
    shown = 0

    with torch.no_grad():
        for encrypted, plain, encrypted_lengths, _ in infer_loader:
            encrypted, plain = encrypted.to(device), plain.to(device)

            # Greedy decoding
            pred_seq = greedy_decode(model, encrypted, encrypted_lengths,
                                     max_len=plain.size(1),
                                     sos_idx=sos_idx, eos_idx=eos_idx,
                                     device=device)
            predictions.append(pred_seq)

            # Métricas
            target_seq = plain[0].cpu().tolist()
            min_len = min(len(pred_seq), len(target_seq))
            greedy_correct += sum(p == t for p, t in zip(pred_seq[:min_len], target_seq[:min_len]))
            greedy_tokens += len(target_seq)
            if pred_seq == target_seq:
                exact_matches += 1

            # Mostrar ejemplos
            if shown < 10:
                encrypted_text = decode_tokens(encrypted[0].cpu().tolist(), dataset.vocab)
                plain_text = decode_tokens(target_seq, dataset.vocab)
                pred_text = decode_tokens(pred_seq, dataset.vocab)

                print(f"Example {shown+1}:")
                print(f"  Plain text    : {plain_text}")
                print(f"  Encrypted text: {encrypted_text}")
                print(f"  Prediction    : {pred_text}")
                print("-" * 50)
                shown += 1

    # Métricas finales
    greedy_acc = greedy_correct / greedy_tokens if greedy_tokens > 0 else 0.0
    exact_acc = exact_matches / len(infer_loader)

    return {
        "predictions": predictions,
        "greedy_acc": greedy_acc,
        "exact_acc": exact_acc
    }
