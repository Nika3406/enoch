import torch
from torch.utils.data import DataLoader
from device import DEVICE

def train_model(model, dataset, epochs=1, batch_size=8, lr=3e-4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.to(DEVICE)
    model.train()

    for epoch in range(epochs):
        for i,(x,y) in enumerate(loader):
            x,y = x.to(DEVICE), y.to(DEVICE)
            _,loss = model(x,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                print(f"Epoch {epoch} Step {i} Loss {loss.item():.4f}")

    torch.save(model.state_dict(),"enoch.pt")
