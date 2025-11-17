import chess
import torch
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# --- 1. CORRECT IMPORTS ---
# These imports match the files in your lichess-bot repository
import chess.engine
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE, Limit

# --- 2. IMPORT YOUR CODE ---
# Add your copied 'src' folder to the Python path
CURRENT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Now, import all your modules from the 'src' folder
try:
    from models.chess_net import ChessNet
    from search.search import choose_best_move
    from board_encoder import encode_board
except ImportError as e:
    print("--- IMPORT ERROR ---")
    print(f"Could not import bot modules from '{SRC_PATH}'")
    print("Please make sure your 'src' folder is copied into this directory.")
    print(f"Error: {e}")
    sys.exit(1)

# --- 3. DOWNLOAD AND LOAD YOUR MODEL (ONCE) ---
print("[MyBot] Initializing ChessNet...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MyBot] Using device: {DEVICE}")

# --- !!! THIS IS YOUR HUGGING FACE REPO INFO !!! ---
HF_REPO_ID = "HamzaAmmar/chesshacks-model"
HF_FILENAME = "best.pt"

# Define a local path to save/cache the weights
WEIGHTS_DIR = Path(CURRENT_DIR) / "downloaded_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# Use the full downloaded path from hf_hub_download
print(f"[MyBot] Downloading weights from Hugging Face repo: {HF_REPO_ID}")
try:
    hf_token = os.environ.get("HF_TOKEN", None) 
    downloaded_model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        cache_dir=str(WEIGHTS_DIR),
        token=hf_token
    )
    print(f"[MyBot] Weights downloaded to {downloaded_model_path}")
except Exception as e:
    print(f"!!! CRITICAL ERROR: Could not download weights from {HF_REPO_ID}")
    print(f"!!! Error: {e}")
    sys.exit(1)

# Load the model weights
MODEL = ChessNet().to(DEVICE)
try:
    # Load from the correct downloaded path
    ckpt = torch.load(downloaded_model_path, map_location=DEVICE)
    
    if "model_state" in ckpt:
        MODEL.load_state_dict(ckpt["model_state"]) 
    else:
        MODEL.load_state_dict(ckpt)
    
    MODEL.eval()
    print(f"[MyBot] Successfully loaded model weights.")
except Exception as e:
    print(f"!!! CRITICAL ERROR: Failed to load model weights from {downloaded_model_path}")
    print(f"!!! Make sure your .pt file matches the ChessNet architecture.")
    print(f"!!! Error: {e}")
    sys.exit(1)


# --- 4. CREATE THE BOT CLASS (The "Engine") ---
# This class name ("MyPyTorchBot") MUST match what you put in config.yml
# It now correctly extends MinimalEngine
class MyPyTorchBot(MinimalEngine):
    """
    This class connects your trained model to the lichess-bot framework.
    """
    def __init__(self, commands, options, stderr, draw_or_resign, game=None, **popen_args):
        super().__init__(commands=commands, options=options, stderr=stderr,
                     draw_or_resign=draw_or_resign, game=game,
                     name="MyPyTorchBot", **popen_args)

    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE) -> chess.engine.PlayResult:
        """
        This is the main function called by lichess-bot when it's our turn.
        It must RETURN a chess.engine.PlayResult object.
        """
        print(f"[MyBot] Searching position: {board.fen()}")

        # Call your own search function from src/search/search.py
        best_move, best_score = choose_best_move(
            board=board,
            model=MODEL,
            device=DEVICE,
            depth=3,
            root_k=20,
            child_k=10,
        )

        if best_move is None:
            print("[MyBot] WARNING: choose_best_move returned None. Picking first legal move.")
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                print("[MyBot] No legal moves found.")
                return chess.engine.PlayResult(None, 0.0) # Return an empty move
            best_move = legal_moves[0]
            best_score = 0.0

        print(f"[MyBot] Move found: {best_move.uci()} (Score: {best_score:.4f})")
        
        # Send the move back to Lichess using the correct return type
        return chess.engine.PlayResult(
            move=best_move,
            ponder=None, # We are not calculating a ponder move
            info={
                # Convert float score to a centipawn score object
                "score": chess.engine.PovScore(chess.engine.Cp(int(best_score * 100)), chess.WHITE),
                "depth": 10
            }
        )