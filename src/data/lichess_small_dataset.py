import io
import chess.pgn
import torch
from torch.utils.data import Dataset

from board_encoder import encode_board
from utils.move_index import move_to_index
from data.lichess_stream_dataset import result_to_value  # reuse helper


class LichessSmallDataset(Dataset):
    """
    Non-streaming dataset for local debugging.
    Builds a finite list of (board, policy_idx, value) from a small HF slice.
    """

    def __init__(self, hf_dataset, max_moves_per_game=20, max_samples=2000):
        self.samples = []

        for row in hf_dataset:
            pgn_text = row.get("pgn") or row.get("PGN") or None
            if pgn_text is None and "moves" in row:
                pgn_text = f"[Result \"*\"]\n\n{row['moves']}"
            if pgn_text is None:
                continue

            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if game is None:
                continue

            value_label = result_to_value(game.headers.get("Result", "*"))

            board = game.board()
            moves = list(game.mainline_moves())[:max_moves_per_game]

            for move in moves:
                encoded = encode_board(board)
                policy_idx = move_to_index(move)

                self.samples.append((
                    torch.tensor(encoded, dtype=torch.float32),
                    torch.tensor(policy_idx, dtype=torch.long),
                    torch.tensor(value_label, dtype=torch.float32),
                ))

                board.push(move)

                if len(self.samples) >= max_samples:
                    return  # stop collecting more

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
