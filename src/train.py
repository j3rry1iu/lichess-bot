import time
import random
from pathlib import Path
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset

from models.chess_net import ChessNet
from data.lichess_stream_dataset import LichessGameStreamDataset


# Directory to save weights. When running on Modal, it's common to mount a persistent volume
# at /weights; if that exists use it. You can also set the environment variable
# MODAL_WEIGHTS_PATH to point to a mounted directory. Otherwise default to repo/local.
if "MODAL_WEIGHTS_PATH" in os.environ:
    WEIGHTS_DIR = Path(os.environ["MODAL_WEIGHTS_PATH"])
else:
    WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"[Train] WEIGHTS_DIR={WEIGHTS_DIR}")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, global_step, avg_loss, name: str):
    ckpt_path = WEIGHTS_DIR / name
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "avg_loss": avg_loss,
        },
        ckpt_path,
    )
    print(f"[Checkpoint] Saved {name} (epoch={epoch}, step={global_step}, loss={avg_loss:.4f})")


def train(
    epochs: int = 3,
    batch_size: int = 256,
    max_moves_per_game: int | None = 80,
    lr: float = 1e-3,
    max_steps_per_epoch: int | None = None,  # Optional limit, None = full dataset
):
    """
    Train ChessNet on Lichess games using a streaming HF dataset.

    - epochs:           number of epochs (full passes through dataset)
    - batch_size:       batch size for DataLoader
    - max_moves_per_game: cap moves taken from each game (for speed)
    - lr:               learning rate
    - max_steps_per_epoch: optional limit on steps per epoch (None = process full dataset)
    """

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    print("[Train] Loading Lichess streaming dataset...")
    hf_train = load_dataset("Lichess/standard-chess-games", streaming=True)["train"]
    
    # Skip shuffle for now to debug
    # hf_train = hf_train.shuffle(seed=42, buffer_size=1000)

    print("[Train] Creating dataset wrapper...")
    dataset = LichessGameStreamDataset(
        hf_dataset=hf_train,
        max_moves_per_game=max_moves_per_game,
    )

    print("[Train] Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
    )

    # 4) Model + optimizer
    print("[Train] Initializing model...")
    model = ChessNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    best_loss = float("inf")

    # Resume support: check for last.pt or best.pt
    start_epoch = 1
    resume_path = WEIGHTS_DIR / "last.pt"
    if not resume_path.exists():
        resume_path = WEIGHTS_DIR / "best.pt"
    
    print(f"[Train] Checking for checkpoint at {resume_path}...")
    if resume_path.exists():
        print(f"[Train] Found checkpoint {resume_path}, resuming...")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_loss = float(ckpt.get("avg_loss", best_loss))
        print(f"[Train] Resumed from epoch {start_epoch-1}, global_step={global_step}")
    else:
        print("[Train] No checkpoint found, starting from scratch.")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        model.train()

        epoch_loss = 0.0
        steps_in_epoch = 0

        print(f"[Epoch {epoch}] Creating data iterator...")
        data_iter = iter(dataloader)
        start_time = time.time()

        step = 0
        while True:
            if step == 0:
                print(f"[Epoch {epoch}] Fetching first batch...")
            
            # Check if we've hit the optional step limit
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                print(f"[Epoch {epoch}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch")
                break
            
            try:
                x, policy_target, value_target = next(data_iter)
            except StopIteration:
                print(f"[Epoch {epoch}] Dataset exhausted after {step} steps")
                break
            
            if step == 0:
                print(f"[Epoch {epoch}] First batch received: x.shape={x.shape}")

            x = x.to(device)                         # (B, 12, 8, 8)
            policy_target = policy_target.to(device) # (B,)
            value_target = value_target.to(device)   # (B,)

            policy_logits, values = model(x)         # (B, NUM_MOVES), (B, 1)
            values = values.squeeze(-1)              # (B,)

            policy_loss = F.cross_entropy(policy_logits, policy_target)
            value_loss = F.mse_loss(values, value_target)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            steps_in_epoch += 1
            step += 1
            epoch_loss += loss.item()

            if global_step % 100 == 0:
                print(f"[Step {global_step}] loss={loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(1, steps_in_epoch)
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch}] avg_loss={avg_epoch_loss:.4f} time={elapsed:.1f}s")

        # Always save "last" checkpoint
        save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss, "last.pt")

        # Save "best" if loss improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss, "best.pt")

        # Optional: per-epoch snapshot
        # save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss, f"epoch_{epoch}.pt")

    # Save plain state_dict for inference
    state_dict_path = WEIGHTS_DIR / "chess_net_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    print(f"[Train] Saved final state_dict to {state_dict_path}")
    print(f"[Train] Best avg epoch loss={best_loss:.4f}")


def main():
    # Train through full dataset for multiple epochs
    train(
        epochs=45,  
        batch_size=256,
        max_moves_per_game=80,
        lr=1e-3,
        max_steps_per_epoch=None,  # No limit - process full dataset
    )


if __name__ == "__main__":
    main()
