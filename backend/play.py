"""
play.py
-------
Play chess against a trained checkpoint interactively.

Usage:
    python play.py checkpoint_iter0050.pt --color white
    python play.py checkpoint_iter0050.pt --color black
    python play.py checkpoint_iter0050.pt --sims 200   # stronger AI

Controls during game:
    - Enter moves in SAN (e.g., "e4", "Nf3", "O-O", "exd5")
    - Type "undo" to take back the last move
    - Type "quit" to exit
    - Type "help" to see legal moves
"""

import argparse
import chess
import torch

from network import ChessNet
from mcts_simple import MCTS


def display_board(board: chess.Board, flip: bool = False):
    """Print the board with coordinates."""
    print()
    if flip:
        print("  " + board.unicode(borders=True, invert_color=True))
    else:
        print("  " + board.unicode(borders=True))
    print()


def get_user_move(board: chess.Board) -> chess.Move | None:
    """Parse user input and return a legal Move, or None to quit."""
    while True:
        try:
            inp = input("Your move: ").strip()
            
            if inp.lower() in ["quit", "exit", "q"]:
                return None
            
            if inp.lower() == "undo":
                if len(board.move_stack) >= 2:
                    board.pop()  # undo AI move
                    board.pop()  # undo user move
                    print("Undone last move.")
                    return "undo"
                else:
                    print("Nothing to undo.")
                    continue
            
            if inp.lower() == "help":
                legal = [board.san(m) for m in board.legal_moves]
                print(f"Legal moves: {', '.join(legal)}")
                continue
            
            # Try parsing as SAN
            move = board.parse_san(inp)
            if move in board.legal_moves:
                return move
            else:
                print(f"Illegal move: {inp}")
                
        except ValueError:
            print(f"Invalid syntax: {inp}. Use SAN (e.g., e4, Nf3, O-O)")


def ai_move(
    board: chess.Board,
    mcts: MCTS,
    show_thinking: bool = True,
) -> chess.Move:
    """Generate AI move using MCTS."""
    if show_thinking:
        print("AI is thinking...", end="", flush=True)
    
    root = mcts.run(board, add_noise=False)
    moves, probs = mcts.get_policy(root)
    
    # Show top-3 candidate moves with visit counts
    if show_thinking:
        top_k = sorted(
            zip(moves, probs, [root.children[m].visit_count for m in moves]),
            key=lambda x: -x[1]
        )[:3]
        print("\r" + " " * 30 + "\r", end="")  # clear "thinking..." line
        for m, p, visits in top_k:
            q = root.children[m].q_value
            print(f"  {board.san(m):8s}  visits={visits:4d}  prob={p:.3f}  Q={q:+.3f}")
    
    best_move = moves[int(probs.argmax())]
    return best_move


def load_checkpoint(path: str):
    """
    Load checkpoint, handling both old and new formats.
    
    Old format: pickled Config object (from early train.py versions)
    New format: num_res_blocks and channels as separate keys
    """
    print(f"Loading checkpoint: {path}")
    
    # Try new format first (weights_only safe mode)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        num_res_blocks = ckpt.get("num_res_blocks", 4)
        channels = ckpt.get("channels", 128)
        iteration = ckpt.get("iteration", "?")
        return ckpt, num_res_blocks, channels, iteration
    except Exception:
        pass  # Fall through to old format handler
    
    # Try old format (pickled Config object - requires weights_only=False)
    try:
        # Create a dummy Config class to satisfy unpickler
        from dataclasses import dataclass
        
        @dataclass
        class Config:
            num_res_blocks: int = 4
            channels: int = 128
            num_sims: int = 100
            num_parallel: int = 64
            c_puct: float = 2.5
            dirichlet_alpha: float = 0.3
            games_per_iter: int = 10
            max_game_moves: int = 200
            replay_buffer_size: int = 20_000
            batch_size: int = 128
            train_steps: int = 200
            lr: float = 1e-3
            weight_decay: float = 1e-4
            num_iterations: int = 50
            device: str = "cuda"
            seed: int = 42
            checkpoint_every: int = 5
            use_compile: bool = False
        
        # Make Config available for unpickling
        import sys
        import __main__
        __main__.Config = Config
        
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        
        if "config" in ckpt and hasattr(ckpt["config"], "num_res_blocks"):
            cfg = ckpt["config"]
            num_res_blocks = cfg.num_res_blocks
            channels = cfg.channels
        else:
            # Final fallback
            num_res_blocks = ckpt.get("num_res_blocks", 4)
            channels = ckpt.get("channels", 128)
        
        iteration = ckpt.get("iteration", "?")
        return ckpt, num_res_blocks, channels, iteration
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using default architecture (4 blocks, 128 channels)")
        # Return minimal valid checkpoint
        net = ChessNet(num_res_blocks=4, channels=128)
        ckpt = {"model": net.state_dict(), "iteration": 0}
        return ckpt, 4, 128, 0


def main():
    parser = argparse.ArgumentParser(description="Play against trained chess AI")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                        help="Your color (default: white)")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per move (default: 100)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Hide AI's candidate moves")
    args = parser.parse_args()

    # Load checkpoint with fallback handling
    ckpt, num_res_blocks, channels, iteration = load_checkpoint(args.checkpoint)
    
    # Build network
    net = ChessNet(num_res_blocks=num_res_blocks, channels=channels)
    net.load_state_dict(ckpt["model"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    print(f"Loaded iteration {iteration} on {device}")
    print(f"Architecture: {num_res_blocks} res-blocks, {channels} channels\n")

    # Set up MCTS
    mcts = MCTS(
        network=net,
        device=device,
        num_sims=args.sims,
        num_parallel=args.sims,  # 1 GPU call per move
        temperature=0.0,         # deterministic best move
    )

    user_is_white = (args.color == "white")
    board = chess.Board()

    print("=" * 60)
    print(f"  You are playing as {args.color.upper()}")
    print(f"  AI strength: {args.sims} simulations per move")
    print("=" * 60)
    print("\nCommands: type move in SAN (e.g. 'e4'), 'undo', 'help', or 'quit'\n")

    display_board(board, flip=not user_is_white)

    move_history = []

    while not board.is_game_over():
        is_user_turn = (board.turn == chess.WHITE) == user_is_white

        if is_user_turn:
            # User's turn
            move = get_user_move(board)
            if move is None:
                print("\nGame aborted.")
                break
            if move == "undo":
                display_board(board, flip=not user_is_white)
                continue
            
            san = board.san(move)
            board.push(move)
            move_history.append(san)
            print(f"You played: {san}")
        
        else:
            # AI's turn
            move = ai_move(board, mcts, show_thinking=not args.no_thinking)
            san = board.san(move)
            board.push(move)
            move_history.append(san)
            print(f"AI played:  {san}\n")
        
        display_board(board, flip=not user_is_white)

    # Game over
    outcome = board.outcome()
    if outcome:
        print("=" * 60)
        if outcome.winner is None:
            print("  DRAW")
        elif outcome.winner == chess.WHITE:
            print("  WHITE WINS" if user_is_white else "  AI (WHITE) WINS")
        else:
            print("  BLACK WINS" if not user_is_white else "  AI (BLACK) WINS")
        print(f"  Termination: {outcome.termination.name}")
        print("=" * 60)
    
    print(f"\nMove history: {' '.join(move_history)}")
    print(f"FEN: {board.fen()}\n")


if __name__ == "__main__":
    main()