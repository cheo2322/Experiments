import torch
from models.seq2seq.seq2seq import Seq2Seq, TransformerEncoder, TransformerDecoder


def decode_tokens(seq, vocab):
    chars = []
    for token in seq:
        ch = vocab[token]
        if ch in ("<sos>", "<pad>"):
            continue
        if ch == "<eos>":
            break
        chars.append(ch)
    return "".join(chars)


def greedy_decode(model, encrypted, max_len, sos_idx, eos_idx, device):
    # Encoder devuelve memory y máscara
    memory, src_key_padding_mask = model.encoder(encrypted)

    # Arrancamos con <sos>
    input_token = torch.tensor([[sos_idx]], device=device)
    decoded = []

    for _ in range(max_len):
        # Decoder recibe input + memory + máscara
        output = model.decoder(
            input_token,
            memory,
            memory_key_padding_mask=src_key_padding_mask
        )
        next_token = output.argmax(-1)[:, -1]

        if next_token.item() == eos_idx:
            break
        decoded.append(next_token.item())

        # El siguiente paso usa el token recién generado
        input_token = torch.cat([input_token, next_token.unsqueeze(0)], dim=1)

    return decoded


def _trim_target(target_row, pad_idx, eos_idx):
    """Quita <sos> inicial y recorta en el primer <eos> o <pad>."""
    target_seq = target_row[1:]  # quitar <sos>
    if eos_idx in target_seq:
        target_seq = target_seq[:target_seq.index(eos_idx)]
    elif pad_idx in target_seq:
        target_seq = target_seq[:target_seq.index(pad_idx)]
    return target_seq


def infer_model(infer_loader, dataset, device, vocab_size, embidding_dim, hidden_dim, output_dir, sos_idx=1, eos_idx=2, pad_idx=0):

    # Rebuild model
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        emb_dim=embidding_dim,
        n_heads=4,
        n_layers=2,
        ff_dim=hidden_dim
    )
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        emb_dim=embidding_dim,
        n_heads=4,
        n_layers=2,
        ff_dim=hidden_dim
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
        for encrypted, plain, _, _ in infer_loader:
            encrypted, plain = encrypted.to(device), plain.to(device)

            # Greedy decoding
            pred_seq = greedy_decode(model, encrypted, max_len=plain.size(1),
                                      sos_idx=sos_idx, eos_idx=eos_idx, device=device)
            predictions.append(pred_seq)

            # Target alineado: sin <sos>, recortado en <eos>/<pad>
            target_seq = _trim_target(plain[0].cpu().tolist(), pad_idx, eos_idx)

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
    exact_acc = exact_matches / len(infer_loader.dataset) if len(infer_loader.dataset) > 0 else 0.0

    return {
        "predictions": predictions,
        "greedy_acc": greedy_acc,
        "exact_acc": exact_acc
    }
