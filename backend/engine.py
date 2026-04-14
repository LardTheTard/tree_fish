import chess

board = chess.Board()

PROMOTION_MAP = {
    "q": chess.QUEEN,
    "r": chess.ROOK,
    "b": chess.BISHOP,
    "n": chess.KNIGHT,
    None: None
}

async def makeResponse(move_dict: dict) -> dict:

    from_square = chess.parse_square(move_dict['from'])
    to_square = chess.parse_square(move_dict['to'])
    promotion = PROMOTION_MAP.get(move_dict['promotion'])
    turn_color = chess.WHITE if move_dict['color'] == 'w' else chess.BLACK

    # CHECK IF MOVE IS A PROMOTION
    move = chess.Move(from_square, to_square)

    board.push(move)

    print(board)

    #Have some processing here
    return