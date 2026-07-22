import torch
from models.seq2seq.seq2seq import PositionwiseModel


def decode_tokens(seq, vocab):
    chars = []
    for token in seq:
        ch = vocab[token]
        if ch == "<pad>":
            continue
        chars.append(ch)
    return "".join(chars)


def infer_model(infer_loader, dataset, device, vocab_size, embidding_dim, hidden_dim, output_dir,
                pad_idx=0, max_len=64):

    model = PositionwiseModel(
        vocab_size=vocab_size,
        emb_dim=embidding_dim,
        n_heads=4,
        n_layers=2,
        ff_dim=hidden_dim,
        max_len=max_len
    ).to(device)

    ckpt_path = f"{output_dir}/best_model.pt"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    predictions = []
    correct, total_tokens, exact_matches, total_examples = 0, 0, 0, 0
    shown = 0

    with torch.no_grad():
        for encrypted, plain, _, _ in infer_loader:
            encrypted, plain = encrypted.to(device), plain.to(device)

            output = model(encrypted)              # (batch, seq_len, vocab_size)
            pred = output.argmax(-1)                # (batch, seq_len)
            predictions.append(pred.cpu().tolist())

            mask = plain != pad_idx
            correct += ((pred == plain) & mask).sum().item()
            total_tokens += mask.sum().item()

            for pred_row, target_row in zip(pred, plain):
                m = target_row != pad_idx
                if torch.equal(pred_row[m], target_row[m]):
                    exact_matches += 1
                total_examples += 1

            if shown < 10:
                encrypted_text = decode_tokens(encrypted[0].cpu().tolist(), dataset.vocab)
                plain_text = decode_tokens(plain[0].cpu().tolist(), dataset.vocab)
                pred_text = decode_tokens(pred[0].cpu().tolist(), dataset.vocab)

                print(f"Example {shown+1}:")
                print(f"  Plain text    : {plain_text}")
                print(f"  Encrypted text: {encrypted_text}")
                print(f"  Prediction    : {pred_text}")
                print("-" * 50)
                shown += 1

    token_acc = correct / total_tokens if total_tokens > 0 else 0.0
    exact_acc = exact_matches / total_examples if total_examples > 0 else 0.0

    return {
        "predictions": predictions,
        "token_acc": token_acc,
        "exact_acc": exact_acc
    }
