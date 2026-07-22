import torch
from models.seq2seq.seq2seq import Seq2Seq, Encoder, Decoder

def decode_tokens(seq, vocab):
    chars = []
    for token in seq:  # token debe ser int
        ch = vocab[token]  # vocab es lista, indexada por int
        if ch == "<sos>":
            continue
        if ch == "<eos>":
            break
        if ch == "<pad>":
            continue
        chars.append(ch)
    return "".join(chars)

def infer_model(infer_loader, dataset, device, vocab_size, embidding_dim, hidden_dim, output_dir):
    # Rebuild model
    encoder = Encoder(vocab_size, embidding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embidding_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

    # Cargar checkpoint
    ckpt_path = f"{output_dir}/best_model.pt"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    predictions = []
    shown = 0
    with torch.no_grad():
        for encrypted, plain, encrypted_lengths, _ in infer_loader:
            plain, encrypted = plain.to(device), encrypted.to(device)
            output = model(encrypted, encrypted_lengths, plain[:, :-1])
            preds = output.argmax(-1)
            predictions.extend(preds.cpu().tolist())

            decoded_batch = preds.cpu().tolist()

            for i, (plain_seq, encrypted_seq, pred_seq) in enumerate(zip(plain, encrypted, decoded_batch)):
                if shown >= 10:
                    break
                encrypted_text = decode_tokens(encrypted_seq, dataset.vocab)  # input
                plain_text = decode_tokens(plain_seq, dataset.vocab)          # target
                pred_text = decode_tokens(pred_seq, dataset.vocab)            # prediction

                print(f"Example {shown+1}:")
                print(f"  Plain text    : {plain_text}")
                print(f"  Encrypted text: {encrypted_text}")
                print(f"  Prediction    : {pred_text}")
                print("-" * 50)
                shown += 1
            if shown >= 10:
                break

    decoded = [[dataset.vocab[token] for token in seq] for seq in predictions]
    return decoded
