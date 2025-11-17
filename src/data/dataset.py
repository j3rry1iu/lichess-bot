# src/data/dataset.py

import torch
from torch.utils.data import Dataset
import chess

from board_encoder import encode_board
from utils.move_index import move_to_index


def result_to_value(result_str: str) -> float:
    """
    Map PGN result string to scalar in [-1, 1] from White's perspective.
    """
    if not result_str:
        return 0.0
    result_str = result_str.strip()
    if result_str == "1-0":
        return 1.0
    if result_str == "0-1":
        return -1.0
    if result_str in ("1/2-1/2", "½-½"):
        return 0.0
    return 0.0


class ChessPositionDataset(Dataset):
    """
    Turns a list of python-chess Game objects into:
      - board encoding (12, 8, 8)
      - policy target: move index (Long)
      - value target: game result scalar (Float in [-1, 1])

    One sample per move in the main line of each game.
    """

    def __init__(self, games, max_moves_per_game=None):
        super().__init__()
        self.samples = []

        for game in games:
            result_str = game.headers.get("Result", "*")
            value_label = result_to_value(result_str)

            board = game.board()
            moves = list(game.mainline_moves())

            if max_moves_per_game is not None:
                moves = moves[:max_moves_per_game]

            for move in moves:
                encoded = encode_board(board)  # (12, 8, 8)
                move_idx = move_to_index(move)

                self.samples.append((encoded, move_idx, value_label))

                board.push(move)

        print(f"[Dataset] Loaded {len(self.samples)} samples from {len(games)} games.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded, move_idx, value_label = self.samples[idx]

        x = torch.tensor(encoded, dtype=torch.float32)          # (12, 8, 8)
        policy_target = torch.tensor(move_idx, dtype=torch.long)
        value_target = torch.tensor(value_label, dtype=torch.float32)

        return x, policy_target, value_target
