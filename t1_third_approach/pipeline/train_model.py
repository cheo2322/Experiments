import torch
import torch.nn as nn
import torch.optim as optim
from models.seq2seq.seq2seq import Seq2Seq, Encoder, Decoder


def train_model(dataloader, vocab_size, emb_dim, hidden_dim,
                lr, epochs, grad_clip, ckpt_every,
                device, output_dir, weight_decay=0.0):

    encoder = Encoder(vocab_size, emb_dim, hidden_dim)
    decoder = Decoder(vocab_size, emb_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

    # Definir loss y optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignorar <pad>
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loop de entrenamiento
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for plain, encrypted, plain_lengths, encrypted_lengths in dataloader:
            plain, encrypted = plain.to(device), encrypted.to(device)

            optimizer.zero_grad()
            # Forward: src=plain, trg=encrypted
            output = model(plain, plain_lengths, encrypted[:, :-1])

            # Calcular loss contra la secuencia target desplazada
            loss = criterion(
                output.reshape(-1, vocab_size),
                encrypted[:, 1:].reshape(-1)
            )
            loss.backward()

            # Clip de gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Guardar checkpoint
        if (epoch + 1) % ckpt_every == 0:
            ckpt_path = f"{output_dir}/model_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint guardado en {ckpt_path}")
