import torch
import torch.nn as nn
import torch.optim as optim
from models.seq2seq.seq2seq import Seq2Seq, Encoder, Decoder
from t1_third_approach.pipeline.eval_model import eval_model


def train_model(train_loader, eval_loader, vocab_size, emb_dim, hidden_dim,
                lr, epochs, grad_clip, device, output_dir, weight_decay=0.0):

    encoder = Encoder(vocab_size, emb_dim, hidden_dim)
    decoder = Decoder(vocab_size, emb_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

    # Definir loss y optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignorar <pad>
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    
    # Loop de entrenamiento
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        
        # Itering over batches
        for plain, encrypted, plain_lengths, _ in train_loader:
            plain, encrypted = plain.to(device), encrypted.to(device)
            optimizer.zero_grad()
            output = model(plain, plain_lengths, encrypted[:, :-1])
            loss = criterion(output.reshape(-1, vocab_size), encrypted[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

        # --- Validation ---
        val_loss, val_acc = eval_model(eval_loader, model, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Guardar mejor modelo según validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = f"{output_dir}/best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Mejor modelo guardado en {ckpt_path}")
