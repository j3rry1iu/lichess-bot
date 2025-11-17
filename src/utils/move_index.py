import chess

BASE_MOVE_SPACE = 4096        # 64 × 64
PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
NUM_PROMOTIONS = 4
NUM_MOVES = BASE_MOVE_SPACE * (NUM_PROMOTIONS + 1)

# Precompute promotion lookup (avoid list.index)
PROMOTION_TO_IDX = {p: i for i, p in enumerate(PROMOTION_PIECES)}

def move_to_index(move: chess.Move) -> int:
    base = move.from_square * 64 + move.to_square

    promo = move.promotion
    if promo is None:
        return base

    promo_idx = PROMOTION_TO_IDX.get(promo, 0)
    return BASE_MOVE_SPACE + base * NUM_PROMOTIONS + promo_idx


def index_to_move(idx: int) -> chess.Move:
    if idx < BASE_MOVE_SPACE:
        return chess.Move(idx // 64, idx % 64)

    tmp = idx - BASE_MOVE_SPACE
    base = tmp // NUM_PROMOTIONS
    promo_idx = tmp % NUM_PROMOTIONS

    from_sq = base // 64
    to_sq = base % 64
    promo_piece = PROMOTION_PIECES[promo_idx]

    return chess.Move(from_sq, to_sq, promotion=promo_piece)