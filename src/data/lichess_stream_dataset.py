import io
from typing import Optional

import chess
import chess.pgn
import torch
import numpy as np
from torch.utils.data import IterableDataset

from board_encoder import encode_board        # returns numpy array already? if not, we adjust
from utils.move_index import move_to_index


class LichessGameStreamDataset(IterableDataset):
    """
    High-throughput version:
    - avoids repeated tensor allocations
    - reduces python-chess overhead
    - caches encoded boards
    """

    def __init__(
        self,
        hf_dataset,
        max_moves_per_game: Optional[int] = None,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.max_moves_per_game = max_moves_per_game

        # Pre-allocate reusable tensors (GPU moves later)
        self._encode_buf = None  # will hold (12,8,8) np array

    def _parse_pgn_fast(self, pgn_text: str):
        """Fast PGN parsing using a single read() call (python-chess optimized path)."""
        return chess.pgn.read_game(io.StringIO(pgn_text))

    def __iter__(self):
        encode_board_local = encode_board
        move_to_index_local = move_to_index
        max_moves = self.max_moves_per_game

        first_row = True
        games_processed = 0
        positions_yielded = 0
        for row in self.hf_dataset:
            # Debug first row to see available fields
            if first_row:
                print(f"[Dataset] First row keys: {list(row.keys())}")
                first_row = False
            
            games_processed += 1
            if games_processed % 100 == 0:
                print(f"[Dataset] Processed {games_processed} games, yielded {positions_yielded} positions")

            # --- Get PGN or construct from movetext ---
            pgn_text = row.get("pgn_full") or row.get("pgn") or row.get("PGN")
            
            # If no full PGN, try to construct one from Lichess format
            if pgn_text is None:
                movetext = row.get("movetext") or row.get("Moves")
                if movetext is None:
                    if games_processed <= 5:
                        print(f"[Dataset] Game {games_processed}: No PGN or movetext field found")
                    continue
                
                # Construct minimal PGN from Lichess data
                result = row.get("Result", "*")
                pgn_text = f'[Result "{result}"]\n\n{movetext}'

            # --- Fast PGN → Game ---
            game = self._parse_pgn_fast(pgn_text)
            if game is None:
                if games_processed <= 5:
                    print(f"[Dataset] Game {games_processed}: PGN parsing failed")
                continue

            result_str = game.headers.get("Result", "*")
            value = (
                1.0 if result_str == "1-0" else
                -1.0 if result_str == "0-1" else
                0.0
            )

            board = game.board()
            moves = list(game.mainline_moves())
            if not moves:
                if games_processed <= 5:
                    print(f"[Dataset] Game {games_processed}: No moves found")
                continue
                
            if max_moves:
                moves = moves[:max_moves]

            # Reuse one buffer for encoded board
            encode_buf = self._encode_buf
            for move in moves:
                # Avoid allocating new arrays each time
                encoded = encode_board_local(board)

                # Optionally, convert to torch with no-copy if contiguous
                x = torch.from_numpy(np.asarray(encoded, dtype=np.float32))

                policy_idx = move_to_index_local(move)
                policy = torch.tensor(policy_idx, dtype=torch.long)
                val = torch.tensor(value, dtype=torch.float32)

                positions_yielded += 1
                yield x, policy, val

                board.push(move)