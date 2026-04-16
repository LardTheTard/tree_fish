from flask import Flask, request
from flask_cors import CORS
import chess

# ------------ ENGINE LOGIC --------------

board = chess.Board()

PROMOTION_MAP = {
    "q": chess.QUEEN,
    "r": chess.ROOK,
    "b": chess.BISHOP,
    "n": chess.KNIGHT,
    None: None
}   

def resetBoard():
    global board
    board = chess.Board()
    return True

def makeResponse(move_dict: dict) -> dict:
    global board
    from_square = chess.parse_square(move_dict['from'])
    to_square = chess.parse_square(move_dict['to'])
    promotion = PROMOTION_MAP.get(move_dict['promotion'])
    turn_color = chess.WHITE if move_dict['color'] == 'w' else chess.BLACK

    if str(board.piece_at(from_square)).lower() == 'p' and (chess.square_rank(to_square) == 0 or chess.square_rank(to_square) == 7):
        move = chess.Move(from_square, to_square, promotion)
    else:
        move = chess.Move(from_square, to_square)

    board.push(move)
    print(board)

    #Have some processing here
    return {
        "status": 200,
        "message": "move made"
    }

# ------------ ENDPOINT LOGIC --------------

app = Flask(__name__)
CORS(app)

@app.route("/send_move", methods=['PUT'])
def response_to_move():
    req = request.get_json()
    response = makeResponse(req)
    return response, 200

@app.route("/send_move/reset_board", methods=['GET'])
def reset_board():
    resetBoard()
    return "OK", 200

if __name__ == "__main__":
    app.run(debug=True)
