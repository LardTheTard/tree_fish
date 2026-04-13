from flask import Flask, request
from flask_cors import CORS
import chess

app = Flask(__name__)
CORS(app)

@app.route("/send_move", methods=['PUT'])
def getText():
    req = request.get_json()
    print(req)
    return req, 200

if __name__ == "__main__":
    app.run(debug=True)
