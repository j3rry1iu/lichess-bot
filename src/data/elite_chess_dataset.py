"""
Dataset wrapper for Q-bert/Elite-Chess-Games with high-Elo games including Magnus Carlsen.
"""
import io
import chess.pgn
import torch
from torch.utils.data import IterableDataset

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_encoder import encode_board
from utils.move_index import move_to_index


class EliteChessDataset(IterableDataset):
    """
    Wraps the Q-bert/Elite-Chess-Games HuggingFace dataset (streaming mode).
    
    Fields in this dataset:
    - 'moves': SAN notation moves (e.g., "1.e4 e5 2.Nf3...")
    - 'white_player', 'black_player': Player names
    - 'white_elo', 'black_elo', 'avg_elo': Elo ratings
    - 'result': Game result
    """

    def __init__(
        self,
        hf_dataset,
        max_moves_per_game: int | None = 80,
        min_avg_elo: int | None = 2600,  # Filter for elite games
        player_filter: str | None = None,  # e.g., "Carlsen" to get only Magnus games
    ):
        """
        Args:
            hf_dataset: The streaming HF dataset
            max_moves_per_game: Cap on moves to process per game
            min_avg_elo: Minimum average Elo to include game
            player_filter: Only include games where this string appears in player name
        """
        self.hf_dataset = hf_dataset
        self.max_moves_per_game = max_moves_per_game
        self.min_avg_elo = min_avg_elo
        self.player_filter = player_filter.lower() if player_filter else None
        
        self.game_count = 0
        self.position_count = 0

    def __iter__(self):
        for row in self.hf_dataset:
            # Apply Elo filter
            avg_elo = row.get("avg_elo")
            if self.min_avg_elo and (avg_elo is None or avg_elo < self.min_avg_elo):
                continue
            
            # Apply player filter
            if self.player_filter:
                white = row.get("white_player", "") or ""
                black = row.get("black_player", "") or ""
                if self.player_filter not in white.lower() and self.player_filter not in black.lower():
                    continue
            
            self.game_count += 1
            
            # Get moves in SAN notation
            moves_str = row.get("moves", "")
            if not moves_str:
                continue
            
            # Parse the game
            try:
                # The 'moves' field contains moves like "1.e4 e5 2.Nf3 Nc6..."
                # Create a minimal PGN for parsing
                pgn_text = f"[Result \"{row.get('result', '*')}\"]\n\n{moves_str}"
                pgn_io = io.StringIO(pgn_text)
                game = chess.pgn.read_game(pgn_io)
                if game is None:
                    continue
            except Exception as e:
                continue

            # Determine winner from result
            result_str = row.get("result", "*")
            if result_str == "1-0":
                winner = chess.WHITE
            elif result_str == "0-1":
                winner = chess.BLACK
            else:
                winner = None  # Draw or unknown

            board = game.board()
            move_count = 0

            for move_obj in game.mainline_moves():
                if self.max_moves_per_game and move_count >= self.max_moves_per_game:
                    break

                # Encode current position
                x = encode_board(board)  # numpy (12,8,8)

                # Get move index
                try:
                    move_idx = move_to_index(move_obj)
                except (ValueError, KeyError):
                    board.push(move_obj)
                    move_count += 1
                    continue

                # Compute value target based on eventual winner
                if winner is None:
                    value_target = 0.0  # draw
                elif board.turn == winner:
                    value_target = 1.0
                else:
                    value_target = -1.0

                # Convert to tensors
                x_tensor = torch.tensor(x, dtype=torch.float32)
                policy_target = torch.tensor(move_idx, dtype=torch.long)
                value_target_tensor = torch.tensor(value_target, dtype=torch.float32)

                yield (x_tensor, policy_target, value_target_tensor)
                
                self.position_count += 1

                board.push(move_obj)
                move_count += 1

            # Print progress every 100 games
            if self.game_count % 100 == 0:
                print(f"[Dataset] Processed {self.game_count} elite games, yielded {self.position_count} positions")
