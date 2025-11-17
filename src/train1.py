import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from models.chess_net import ChessNet
from data.local_pgn_data import LocalPGNDataset


# -------------------------
# TRAINING LOOP
# -------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_batches = 0

    for batch_idx, (x, policy_target, value_target) in enumerate(loader):
        x = x.to(device)                                 # (B, 12, 8, 8)
        policy_target = policy_target.to(device)         # (B,)
        value_target = value_target.to(device).float()   # (B,)

        optimizer.zero_grad()

        policy_logits, value_pred = model(x)
        value_pred = value_pred.squeeze(-1)              # (B,)

        # Losses
        policy_loss = F.cross_entropy(policy_logits, policy_target)
        value_loss = F.mse_loss(value_pred, value_target)

        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if batch_idx % 10 == 0:
            print(f"[Batch {batch_idx}] loss={loss.item():.4f}")

    return total_loss / max(1, total_batches)


# -------------------------
# MAIN TRAIN FUNCTION
# -------------------------
def main():

    # ---------------------------------------------------------
    # 1️⃣ LOAD YOUR SINGLE PGN FILE HERE
    # ---------------------------------------------------------
    # ⇒ CHANGE ONLY THIS LINE
    pgn_path = "data/localGame.pgn"
    # ---------------------------------------------------------

    print(f"Loading dataset from: {pgn_path}")

    dataset = LocalPGNDataset(pgn_path, max_moves=60)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------------------------------------
    # 2️⃣ LOAD MODEL
    # ---------------------------------------------------------
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---------------------------------------------------------
    # 3️⃣ TRAIN FOR A FEW SMALL EPOCHS
    # ---------------------------------------------------------
    epochs = 3   # small number for local testing
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, device)
        print(f"[Epoch {epoch}] avg_loss = {avg_loss:.4f}")

    # ---------------------------------------------------------
    # 4️⃣ SAVE WEIGHTS
    # ---------------------------------------------------------
    save_path = "weights/local_debug.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")


if __name__ == "__main__":
    main()
