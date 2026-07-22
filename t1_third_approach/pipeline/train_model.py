import torch
import torch.nn as nn
import torch.optim as optim

from graphics.graphic import plot_metrics
from models.seq2seq.seq2seq import Seq2Seq, Encoder, Decoder
from t1_third_approach.pipeline.eval_model import compute_accuracy, eval_model


def train_model(loaders, vocab_size, emb_dim, hidden_dim,
                lr, epochs, grad_clip, device, output_dir, weight_decay=0.0,
                print_every = 1, loss_threshold = 2.0, acc_threshold = 0.5):

    encoder = Encoder(vocab_size, emb_dim, hidden_dim)
    decoder = Decoder(vocab_size, emb_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Updates the learning rate based on the validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    best_eval_loss = float('inf')
    best_eval_acc = 0.0
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for encrypted, plain, encrypted_lengths, _ in loaders[0]:
            encrypted, plain = encrypted.to(device), plain.to(device)
            optimizer.zero_grad()

            # Forward: input = encrypted, target = plain
            output = model(encrypted, encrypted_lengths, plain[:, :-1])
            
            # Fails if vocabulary does not match the model's output size
            assert output.size(-1) == vocab_size, f"{output.size(-1)} != {vocab_size}"

            # Loss: comparar contra texto plano
            loss = criterion(output.reshape(-1, vocab_size), plain[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            # Accuracy: comparar contra texto plano
            correct, tokens = compute_accuracy(output, plain[:, 1:], pad_idx=0)
            total_loss += loss.item() * tokens
            total_correct += correct
            total_tokens += tokens


        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        train_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

        # Validation
        metrics = eval_model(loaders[1], model, criterion, device)
        val_loss = metrics["loss"]
        val_acc = metrics["teacher_acc"]
        greedy_acc = metrics["greedy_acc"]
        exact_acc = metrics["exact_acc"]

        scheduler.step(val_loss)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if epoch % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best according to validation loss and accuracy thresholds
        # if val_loss < loss_threshold and val_acc > acc_threshold and val_loss < best_eval_loss and val_acc > best_eval_acc:
        if val_loss < best_eval_loss and val_acc > best_eval_acc:
            best_eval_loss = val_loss
            best_eval_acc = val_acc
            ckpt_path = f"{output_dir}/best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}. Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, (Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f})")

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, greedy_accs=greedy_acc, exact_accs=exact_acc)
