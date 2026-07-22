import torch
import torch.nn as nn

K = 42
emb_dim = 32

model = nn.Sequential(
    nn.Embedding(256, emb_dim),
    nn.Linear(emb_dim, 256)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for step in range(2000):
    x = torch.randint(0, 256, (64,))
    y = x ^ K
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        acc = (logits.argmax(-1) == y).float().mean().item()
        print(f"step {step}, loss {loss.item():.4f}, acc {acc:.4f}")