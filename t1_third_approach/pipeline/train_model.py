import torch
import torch.nn as nn
import torch.optim as optim
from models.seq2seq.seq2seq import Seq2Seq, Encoder, Decoder
from t1_third_approach.pipeline.eval_model import eval_model


def train_model(loaders, vocab_size, emb_dim, hidden_dim,
                lr, epochs, grad_clip, device, output_dir, weight_decay=0.0,
                print_every = 1, loss_threshold = 2.0, acc_threshold = 0.5):

    encoder = Encoder(vocab_size, emb_dim, hidden_dim)
    decoder = Decoder(vocab_size, emb_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        
        # Training
        model.train()
        total_loss = 0
        
        # Itering over batches
        for plain, encrypted, plain_lengths, _ in loaders[0]:
            plain, encrypted = plain.to(device), encrypted.to(device)
            optimizer.zero_grad()
            output = model(plain, plain_lengths, encrypted[:, :-1])
            loss = criterion(output.reshape(-1, vocab_size), encrypted[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loaders[0])

        # Validation
        val_loss, val_acc = eval_model(loaders[1], model, criterion, device)
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best according to validation loss and accuracy thresholds
        if val_loss < loss_threshold and val_acc > acc_threshold:
            ckpt_path = f"{output_dir}/best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Modelo guardado en {ckpt_path}. Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, (Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f})")
