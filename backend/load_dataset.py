import msgpack

def load_positions(filepath):
    """Stream positions from a msgpack file."""
    with open(filepath, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for record in unpacker:
            yield record

# Example
for record in load_positions('train-00000-of-01024.msgpack'):
    fen = record['fen']
    moves = record['moves']
    for move, eval in moves.items():
        print(f"{move}: win_prob={eval['win_prob']:.3f}, mate={eval['mate']}")
    print(fen)
    break
