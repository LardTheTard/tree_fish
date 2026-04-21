import msgpack
import os

directory = r'C:\Users\login\tree_fish\tree_fish\backend\data'

def load_positions(filepath):
    """Stream positions from a msgpack file."""
    with open(filepath, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for record in unpacker:
            yield record

num = 0
# Example

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    for idx, record in enumerate(load_positions(filepath)):
        fen = record['fen']
        moves = record['moves']
        
        num = max(num ,idx)
    print(num)