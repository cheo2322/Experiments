import torch
from models.seq2seq.seq2seq import Seq2Seq, Encoder, Decoder

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
    print(f"Modelo cargado desde {ckpt_path}")

    predictions = []
    shown = 0
    with torch.no_grad():
        for plain, encrypted, plain_lengths, _ in infer_loader:
            plain, encrypted = plain.to(device), encrypted.to(device)
            output = model(plain, plain_lengths, encrypted[:, :-1])
            preds = output.argmax(-1)
            predictions.extend(preds.cpu().tolist())

            decoded_batch = [[dataset.vocab[token] for token in seq] for seq in preds.cpu().tolist()]

            for i, (plain_seq, encrypted_seq, pred_seq) in enumerate(zip(plain, encrypted, decoded_batch)):
                if shown >= 10:
                    break
                plain_text = " ".join([dataset.vocab[token.item()] for token in plain_seq])
                encrypted_text = " ".join([dataset.vocab[token.item()] for token in encrypted_seq])
                pred_text = " ".join(pred_seq)

                print(f"Ejemplo {shown+1}:")
                print(f"  Texto plano   : {plain_text}")
                print(f"  Texto cifrado : {encrypted_text}")
                print(f"  Predicción    : {pred_text}")
                print("-" * 50)
                shown += 1
            if shown >= 10:
                break

    decoded = [[dataset.vocab[token] for token in seq] for seq in predictions]
    return decoded
