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
    with torch.no_grad():
        for plain, encrypted, plain_lengths, _ in infer_loader:
            plain, encrypted = plain.to(device), encrypted.to(device)
            output = model(plain, plain_lengths, encrypted[:, :-1])
            preds = output.argmax(-1)
            predictions.extend(preds.cpu().tolist())

    # Decodificar tokens → texto
    # Aquí cambiamos .itos por acceso directo a la lista
    decoded = [[dataset.vocab[token] for token in seq] for seq in predictions]

    print("Resultados de inferencia:")
    for i, seq in enumerate(decoded[:10]):  # mostrar primeros 10
        print(f"Ejemplo {i+1}: {' '.join(seq)}")

    return decoded
