"""
board_encoder.py
----------------
Converts a python-chess Board into the input tensor expected by the network.

Plane layout (18 planes total, each 8×8):
  0–5   : White pieces  (P, N, B, R, Q, K)
  6–11  : Black pieces  (P, N, B, R, Q, K)
  12    : Side to move  (1.0 = white, 0.0 = black, filled)
  13    : White kingside castling right
  14    : White queenside castling right
  15    : Black kingside castling right
  16    : Black queenside castling right
  17    : En-passant file (column of the ep square, normalised to [0,1])
"""

import chess
import torch
import numpy as np

# Piece type → plane offset (same for both colors, color selects 0-5 vs 6-11)
PIECE_TO_IDX = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

NUM_PLANES = 18   # total input channels
BOARD_SIZE = 8


def board_to_tensor(board: chess.Board, device: torch.device | None = None) -> torch.Tensor:
    """
    Returns a (18, 8, 8) float32 tensor representing the board state.

    Square ordering: chess.A1 = 0 (bottom-left), chess.H8 = 63 (top-right).
    We map square index s → row = s // 8, col = s % 8 so that row 0 is rank 1
    and row 7 is rank 8 (consistent with the network seeing the board from
    White's perspective regardless of side to move).
    """
    planes = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Piece planes
    for sq, piece in board.piece_map().items():
        row, col = sq // 8, sq % 8
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane_idx = color_offset + PIECE_TO_IDX[piece.piece_type]
        planes[plane_idx, row, col] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    # En passant file
    if board.ep_square is not None:
        ep_col = board.ep_square % 8
        planes[17, :, :] = ep_col / 7.0  # normalise to [0, 1]

    tensor = torch.from_numpy(planes)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# Move encoding for the policy head
# ---------------------------------------------------------------------------
# AlphaZero encodes 4672 possible actions = 73 "move types" × 64 squares.
# Move types:
#   0–55  : Queen-style slides (8 directions × 7 distances)
#   56–63 : Knight moves (8 L-shapes)
#   64–72 : Underpromotions (3 piece types × 3 directions: left/straight/right)
#            Knight promotion is handled via knight-move encoding above.
#
# Here we use a simplified flat encoding keyed by (from_sq, to_sq, promotion)
# which covers all legal moves in practice. We map each legal move to a unique
# integer index in [0, NUM_ACTIONS).

NUM_ACTIONS = 64 * 73  # 4672 — matches AlphaZero


# Direction vectors for queen slides and knight jumps
_QUEEN_DIRS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),   # rook
    (1, 1), (1, -1), (-1, 1), (-1, -1), # bishop
]
_KNIGHT_DELTAS = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]
PROMO_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]  # not queen (queen is default)
_PROMO_DIRS = [(-1, 1), (0, 1), (1, 1)]  # left-diag, straight, right-diag (from White's POV)


def _build_move_index():
    """Pre-compute move → action_index mapping for all geometrically valid moves."""
    move_to_idx: dict[tuple, int] = {}

    for sq in range(64):
        row, col = sq // 8, sq % 8

        # Queen-style slides
        for dir_idx, (dr, dc) in enumerate(_QUEEN_DIRS):
            for dist in range(1, 8):
                nr, nc = row + dr * dist, col + dc * dist
                if 0 <= nr < 8 and 0 <= nc < 8:
                    to_sq = nr * 8 + nc
                    plane = dir_idx * 7 + (dist - 1)  # 0-55
                    idx = sq * 73 + plane
                    move_to_idx[(sq, to_sq, None)] = idx

        # Knight moves
        for k_idx, (dr, dc) in enumerate(_KNIGHT_DELTAS):
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                to_sq = nr * 8 + nc
                plane = 56 + k_idx  # 56-63
                idx = sq * 73 + plane
                move_to_idx[(sq, to_sq, None)] = idx

        # Underpromotions (pawn on rank 7 → rank 8, White's POV)
        for promo_idx, piece in enumerate(_UNDER_PROMO_PIECES):
            for dir_idx, (dc_off, dr_off) in enumerate(_PROMO_DIRS):
                nr, nc = row + dr_off, col + dc_off
                if 0 <= nr < 8 and 0 <= nc < 8:
                    to_sq = nr * 8 + nc
                    plane = 64 + promo_idx * 3 + dir_idx  # 64-72
                    idx = sq * 73 + plane
                    # Only override if not already mapped (queen slide takes priority)
                    key = (sq, to_sq, piece)
                    if key not in move_to_idx:
                        move_to_idx[key] = idx

    return move_to_idx


_MOVE_TO_IDX = _build_move_index()


def move_to_action(move: chess.Move) -> int:
    """Convert a chess.Move to a flat action index in [0, NUM_ACTIONS)."""
    promo = move.promotion
    if promo == chess.QUEEN:
        promo = None  # queen promotion handled as regular queen slide
    key = (move.from_square, move.to_square, promo)
    if key not in _MOVE_TO_IDX:
        # Fallback: use raw (from, to) without promotion tag
        key = (move.from_square, move.to_square, None)
    return _MOVE_TO_IDX.get(key, 0)


def legal_moves_mask(board: chess.Board, device: torch.device | None = None) -> torch.Tensor:
    """
    Returns a boolean mask of shape (NUM_ACTIONS,) with True at every index
    corresponding to a legal move in the current position.
    """
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_action(move)] = True
    if device is not None:
        mask = mask.to(device)
    return mask


def action_to_move(action_idx: int, board: chess.Board) -> chess.Move | None:
    """
    Reverse-map an action index back to a chess.Move by checking legal moves.
    Returns None if no legal move matches.
    """
    for move in board.legal_moves:
        if move_to_action(move) == action_idx:
            return move
    return None

def main():
    for entry in _MOVE_TO_IDX:
        if entry[2] == 

if __name__ == '__main__':
    main()