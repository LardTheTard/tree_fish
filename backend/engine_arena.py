import chess
import torch
import random

from network import ChessNet
from mcts import MCTS

from tqdm import tqdm

CHECKPOINT1 = r'C:\Users\ZhaoLo\chess\backend\dataset_trained_40iter.pt'
CHECKPOINT2 = r'C:\Users\ZhaoLo\chess\backend\checkpoint_iter2000.pt'
NUM_SIMS = 100
NUM_SIMS_2 = 100
SHOW_THINKING = False
NUM_GAMES = 10

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
            seed: int = 72
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
    # Load checkpoint with fallback handling
    ckpt, num_res_blocks, channels, iteration = load_checkpoint(CHECKPOINT1)
    
    # Build network
    net = ChessNet(num_res_blocks=num_res_blocks, channels=channels)
    net.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    print(f"1: Loaded iteration {iteration} on {device}")
    print(f"1: Architecture: {num_res_blocks} res-blocks, {channels} channels\n")

    # Set up MCTS
    mcts = MCTS(
        network=net,
        device=device,
        num_sims=NUM_SIMS,
        temperature=0.0,         # deterministic best move
    )

    # Load checkpoint with fallback handling
    ckpt, num_res_blocks, channels, iteration = load_checkpoint(CHECKPOINT2)
    
    # Build network
    net_2 = ChessNet(num_res_blocks=num_res_blocks, channels=channels)
    net_2.load_state_dict(ckpt["model"])
    net_2 = net_2.to(device)
    net_2.eval()
    print(f"2: Loaded iteration {iteration} on {device}")
    print(f"2: Architecture: {num_res_blocks} res-blocks, {channels} channels\n")

    # Set up MCTS
    mcts_2 = MCTS(
        network=net_2,
        device=device,
        num_sims=NUM_SIMS_2,
        temperature=0.0,         # deterministic best move
    )
    
    num_engine1_wins = 0
    num_draws = 0

    for game_idx in tqdm(range(NUM_GAMES)):
        board = chess.Board()

        if game_idx % 2 == 0:
            user_is_white = True
        else:
            user_is_white = False

        while not board.is_game_over():
            is_user_turn = (board.turn == chess.WHITE) == user_is_white

            if is_user_turn:
                move = ai_move(board, mcts, show_thinking=SHOW_THINKING)
                board.push(move)
            
            else:
                move = ai_move(board, mcts_2, show_thinking=SHOW_THINKING)
                board.push(move)


        # Game over
        outcome = board.outcome()
        if outcome:
            if outcome.winner is None:
                num_draws += 1
            elif outcome.winner == chess.WHITE and user_is_white:
                num_engine1_wins += 1
            elif outcome.winner == chess.BLACK and not user_is_white:
                num_engine1_wins += 1
                
    
    print("Network 1 won", num_engine1_wins, "times")
    print("Network 1 drew", num_draws, "times")
    print("Network 1 lost", NUM_GAMES - num_engine1_wins - num_draws, "times")


if __name__ == "__main__":
    main()