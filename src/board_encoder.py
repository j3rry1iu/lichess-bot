import chess
import numpy as np

PIECE_TO_CHANNEL = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,

    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

def encode_board(board: chess.Board) -> np.ndarray:
    # New, faster version
    arr = np.zeros((12, 8, 8), dtype=np.int8)

    # Loop squares 0..63 – very fast, contiguous
    for sq in range(64):
        p = board.piece_at(sq)
        if p is None:
            continue

        chan = PIECE_TO_CHANNEL[(p.piece_type, p.color)]

        rank = sq // 8           # rank 0..7
        file = sq % 8            # file 0..7

        arr[chan, 7 - rank, file] = 1  # flip rank so rank 8 is at top

    return arr

if __name__ == "__main__":
    board = chess.Board()
    print(board)
    encoded = encode_board(board)
    print(encoded)
