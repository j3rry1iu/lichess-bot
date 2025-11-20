import torch
import chess
from utils.move_index import move_to_index
from board_encoder import encode_board
from models.chess_net import ChessNet

# --- HELPER: Cached Model Prediction ---
def get_model_output(board: chess.Board, model: ChessNet, device: str, cache: dict):
    """
    Checks if the board position has already been evaluated.
    If yes, returns the cached (policy, value).
    If no, runs the model, caches the result, and returns it.
    """
    board_fen = board.fen() 
    if board_fen in cache:
        return cache[board_fen]
    
    encoded = encode_board(board)
    # Ensure dtype matches your model (Float16/Half if you optimized, else Float32)
    x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)

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
    return value if board.turn == chess.WHITE else -value

# --- QUIESCENCE SEARCH (Fixes Horizon Effect) ---
def quiescence_search(board, alpha, beta, model, device, cache):
    """
    Searches only captures to settle dynamic positions.
    """
    # 1. Stand-pat score (Evaluation if we do nothing)
    stand_pat = evaluate_position(board, model, device, cache)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    # 2. Get Policy for move ordering
    policy_logits, _ = get_model_output(board, model, device, cache)
    
    # 3. Filter for CAPTURES only
    capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
    
    # Score and sort captures
    scored_captures = []
    for move in capture_moves:
        idx = move_to_index(move)
        score = policy_logits[0, idx].item()
        scored_captures.append((score, move))
    scored_captures.sort(reverse=True, key=lambda x: x[0])
    
    for _, move in scored_captures:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, model, device, cache)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
            
    return alpha

# --- MINIMAX ---
def minimax(board, depth, alpha, beta, model, device="cpu", k_moves=10, cache=None):
    if board.is_game_over():
        return evaluate_position(board, model, device, cache)

    # If depth runs out, enter Quiescence Search instead of evaluating immediately
    if depth == 0:
        return quiescence_search(board, alpha, beta, model, device, cache)

    policy_logits, _ = get_model_output(board, model, device, cache)
    ordered_moves = order_moves(board, policy_logits, k=k_moves)

    if not ordered_moves: # No legal moves (stalemate/checkmate handled by game_over check generally, but safety first)
         return evaluate_position(board, model, device, cache)

    # Maximize for current player (Negamax logic is simpler, but keeping your Minimax structure)
    if board.turn == chess.WHITE:
        max_eval = -float("inf")
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, model, device, k_moves, cache)
            board.pop()

            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, model, device, k_moves, cache)
            board.pop()

            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def choose_best_move(board: chess.Board, model: ChessNet, device="cpu", depth: int = 3, root_k: int = 20, child_k: int = 10):
    # Fresh cache for every search
    cache = {}
    
    policy_logits, _ = get_model_output(board, model, device, cache)
    root_moves = order_moves(board, policy_logits, k=root_k)

    best_move = None
    
    if board.turn == chess.WHITE:
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")
        
        for move in root_moves:
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, model, device, k_moves=child_k, cache=cache)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            
            # Update Alpha at root
            alpha = max(alpha, score)
            
    else: # BLACK
        best_score = float("inf")
        alpha = -float("inf")
        beta = float("inf")

        for move in root_moves:
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, model, device, k_moves=child_k, cache=cache)
            board.pop()

            if score < best_score:
                best_score = score
                best_move = move
            
            # Update Beta at root
            beta = min(beta, score)

    return best_move, best_score