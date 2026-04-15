from flask import Flask, request
from flask_cors import CORS
from engine import makeResponse, resetBoard

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
