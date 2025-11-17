import torch
import chess
from utils.move_index import move_to_index
from board_encoder import encode_board
from models.chess_net import ChessNet


def order_moves(board: chess.Board, policy_logits: torch.Tensor, k:int | None = None):
    legal_moves = list(board.legal_moves)
    scored = []

    for move in legal_moves:
        idx = move_to_index(move)
        score = policy_logits[idx].item()
        scored.append((score, move))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    ordered = [m for _, m in scored]

    if k is not None and len(ordered) > k:
        ordered = ordered[:k]
    return ordered

def evaluate_position(board: chess.Board, model: ChessNet, device="cpu") -> float:
    encoded = encode_board(board)
    x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        _, value = model(x)
        value = value.item()
    return value if board.turn == chess.WHITE else -value


def minimax(board:chess.Board, depth: int, alpha:float, beta:float, model:ChessNet, device="cpu", k_moves:int=10) -> float:
    if depth == 0 or board.is_game_over():
        return evaluate_position(board, model, device)

    encoded = encode_board(board)
    x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = model(x)
        policy_logits = policy_logits[0]
    
    ordered_moves = order_moves(board, policy_logits, k=k_moves)

    if board.turn == chess.WHITE:
        max_eval = -float("inf")
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth-1, alpha, beta, model, device, k_moves)
            board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
            if eval_score > alpha:
                alpha = eval_score
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth-1, alpha, beta, model, device, k_moves)
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
            if eval_score < beta:
                beta = eval_score
            if beta <= alpha:
                break
        return min_eval
    

def choose_best_move(board:chess.Board, model:ChessNet, device="cpu", depth:int=3, root_k:int=20, child_k: int=10):
    encoded = encode_board(board)
    x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, _ = model(x)
        policy_logits = policy_logits[0]
    
    root_moves = order_moves(board, policy_logits, k=root_k)

    best_move = None
    if board.turn == chess.WHITE:
        best_score = -float("inf")
        for move in root_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha=-float("inf"), beta=float("inf"), model=model, device=device, k_moves=child_k)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
    else:
        best_score = float("inf")
        for move in root_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha=-float("inf"), beta=float("inf"), model=model, device=device, k_moves=child_k)
            board.pop()

            if score < best_score:
                best_score = score
                best_move = move
    return best_move, best_score