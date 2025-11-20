import torch
import chess
import time
from utils.move_index import move_to_index
from board_encoder import encode_board
from models.chess_net import ChessNet

class SearchTimeout(Exception):
    pass

# --- HELPER: Cached Model Prediction ---
def get_model_output(board: chess.Board, model: ChessNet, device: str, cache: dict):
    board_fen = board.fen() 
    if board_fen in cache:
        return cache[board_fen]
    
    encoded = encode_board(board)
    # Use Float16 if CUDA, else Float32
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    x = torch.tensor(encoded, dtype=dtype).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = model(x)
        policy_logits = policy_logits.cpu()
        value = value.item()
    
    cache[board_fen] = (policy_logits, value)
    return policy_logits, value

def order_moves(board: chess.Board, policy_logits: torch.Tensor, k: int | None = None):
    legal_moves = list(board.legal_moves)
    scored = []
    for move in legal_moves:
        idx = move_to_index(move)
        score = policy_logits[0, idx].item()
        scored.append((score, move))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    ordered = [m for _, m in scored]
    if k is not None and len(ordered) > k:
        ordered = ordered[:k]
    return ordered

def evaluate_position(board: chess.Board, model: ChessNet, device="cpu", cache=None) -> float:
    _, value = get_model_output(board, model, device, cache)
    # IMPORTANT FIX:
    # The model returns +1 (White Win) to -1 (Black Win).
    # Negamax requires the score to be relative to the side to move.
    # If White to move: return value (Positive = Good).
    # If Black to move: return -value (Negative value becomes Positive = Good).
    return value if board.turn == chess.WHITE else -value

# --- TIME CHECK ---
def check_time(end_time, nodes_visited):
    if nodes_visited is not None:
        nodes_visited[0] += 1
        if end_time is not None and nodes_visited[0] % 1000 == 0:
            if time.time() > end_time:
                raise SearchTimeout()

# --- QUIESCENCE SEARCH (Negamax) ---
def quiescence_search(board, alpha, beta, model, device, cache, end_time, nodes_visited):
    check_time(end_time, nodes_visited)
    
    # 1. Stand-pat score
    stand_pat = evaluate_position(board, model, device, cache)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    # 2. Search Captures
    policy_logits, _ = get_model_output(board, model, device, cache)
    capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
    
    scored_captures = []
    for move in capture_moves:
        idx = move_to_index(move)
        score = policy_logits[0, idx].item()
        scored_captures.append((score, move))
    scored_captures.sort(reverse=True, key=lambda x: x[0])
    
    for _, move in scored_captures:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, model, device, cache, end_time, nodes_visited)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
            
    return alpha

# --- NEGAMAX SEARCH ---
def negamax(board, depth, alpha, beta, model, device="cpu", k_moves=10, cache=None, end_time=None, nodes_visited=None):
    check_time(end_time, nodes_visited)

    if board.is_game_over():
        # If game over, return evaluation (e.g. +1 for mate, 0 for draw)
        # evaluate_position handles the perspective correctly.
        return evaluate_position(board, model, device, cache)

    if depth == 0:
        return quiescence_search(board, alpha, beta, model, device, cache, end_time, nodes_visited)

    policy_logits, _ = get_model_output(board, model, device, cache)
    ordered_moves = order_moves(board, policy_logits, k=k_moves)

    if not ordered_moves:
         return evaluate_position(board, model, device, cache)

    max_score = -float("inf")

    for move in ordered_moves:
        board.push(move)
        # Recursive Negamax: Swap alpha/beta, negate score
        score = -negamax(board, depth - 1, -beta, -alpha, model, device, k_moves, cache, end_time, nodes_visited)
        board.pop()

        if score > max_score:
            max_score = score
        
        alpha = max(alpha, score)
        if alpha >= beta:
            break # Pruning
            
    return max_score

# --- ROOT SEARCH ---
def choose_best_move(board: chess.Board, model: ChessNet, device="cpu", depth: int = 3, root_k: int = 20, child_k: int = 10, end_time=None):
    cache = {}
    nodes_visited = [0]
    
    policy_logits, _ = get_model_output(board, model, device, cache)
    root_moves = order_moves(board, policy_logits, k=root_k)

    best_move = None
    best_score = -float("inf")
    alpha = -float("inf")
    beta = float("inf")

    # At the root, we just maximize 'score' (which represents "Good for Us")
    for move in root_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, model, device, child_k, cache, end_time, nodes_visited)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        
        alpha = max(alpha, score)

    # Convert Relative Score back to Absolute Score (White-Centric) for UI display
    # If it's Black's turn, best_score is positive (good for black).
    # We negate it to show -0.9 (White losing) in the logs.
    final_display_score = best_score if board.turn == chess.WHITE else -best_score

    return best_move, final_display_score