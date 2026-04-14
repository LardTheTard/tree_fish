from flask import Flask, request
from flask_cors import CORS
from engine import makeResponse

app = Flask(__name__)
CORS(app)

@app.route("/send_move", methods=['PUT'])
async def response_to_move():
    req = request.get_json()

    response = await makeResponse(req)

    return response, 200

if __name__ == "__main__":
    app.run(debug=True)
