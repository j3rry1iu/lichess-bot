# src/main.py

from pathlib import Path
import torch
import chess

from models.chess_net import ChessNet
from search.search import choose_best_move

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessNet().to(device)

    # Load "best" checkpoint
    ckpt = torch.load(WEIGHTS_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    board = chess.Board()  # or some test FEN

    best_move, best_score = choose_best_move(
        board,
        model,
        device=device,
        depth=3,
        root_k=20,
        child_k=10,
    )

    print("Best move:", best_move, "Score:", best_score)
