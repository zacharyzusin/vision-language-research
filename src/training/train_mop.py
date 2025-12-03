import torch
from torch.utils.data import DataLoader
from src.datasets.inat_dataset import get_inat2018
from src.models.mop_clip import MixturePromptCLIP
from tqdm import tqdm

def train():
    device = "cuda"

    dataset_root = "data/iNat2018"
    train_ds = get_inat2018(dataset_root, "train")
    val_ds = get_inat2018(dataset_root, "val")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        num_classes=num_classes,
        K=config["model"]["K"],
        ctx_len=8
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(20):
        model.train()
        total_loss = 0

        for imgs, targets in tqdm(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)

            loss = model(imgs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} loss = {total_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    train()
