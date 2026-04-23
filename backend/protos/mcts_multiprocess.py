"""
mcts_multiprocess.py
--------------------
Root parallelization: Run independent MCTS searches in parallel workers and merge results.

This approach is simpler than tree parallelization but very effective:
  - Each worker runs independent MCTS on the same position
  - Results are merged by summing visit counts
  - Near-linear speedup with number of workers
"""

from __future__ import annotations

import math
import chess
import numpy as np
import torch
import multiprocessing as mp
from typing import Optional

from board_encoder import board_to_tensor, legal_moves_mask, move_to_action, NUM_ACTIONS
from network import ChessNet


class MCTSNode:
    """A node in the search tree."""
    
    def __init__(
        self,
        board: chess.Board,
        parent: Optional[MCTSNode] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0,
    ):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: dict[chess.Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.distance = 2**32
        self.is_expanded = False
    
    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u


def _worker_search(
    worker_id: int,
    fen: str,
    num_sims: int,
    model_state: dict,
    device_name: str,
    c_puct: float,
    add_noise: bool,
    dirichlet_alpha: float,
    batch_size: int,
) -> dict[str, int]:
    """
    Worker function: run MCTS and return visit counts.
    
    Returns dict mapping move UCI strings to visit counts.
    """
    # Setup device and model in worker
    device = torch.device(device_name)
    net = ChessNet(num_res_blocks=4, channels=128).to(device)
    net.load_state_dict(model_state)
    net.eval()
    
    # Import here to avoid pickling issues
    from mcts_parallel import ParallelMCTS
    
    # Run MCTS with worker-specific random seed
    np.random.seed(worker_id)
    board = chess.Board(fen)
    
    mcts = ParallelMCTS(
        network=net,
        device=device,
        num_sims=num_sims,
        batch_size=batch_size,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        temperature=1.0,
    )
    
    root = mcts.run(board, add_noise=add_noise)
    
    # Extract visit counts
    visit_counts = {
        move.uci(): child.visit_count
        for move, child in root.children.items()
    }
    
    return visit_counts


class MultiprocessMCTS:
    """
    Root parallelization MCTS: run independent searches in parallel and merge.
    
    Args:
        network: ChessNet policy-value network
        device: torch device for main process
        num_workers: number of parallel worker processes
        sims_per_worker: simulations per worker
        batch_size: batch size for each worker's batched MCTS
        c_puct: exploration constant
        dirichlet_alpha: Dirichlet noise parameter
        temperature: move sampling temperature
    """
    
    def __init__(
        self,
        network: ChessNet,
        device: torch.device,
        num_workers: int = 4,
        sims_per_worker: int = 200,
        batch_size: int = 16,
        c_puct: float = 2.5,
        dirichlet_alpha: float = 0.3,
        temperature: float = 1.0,
    ):
        self.net = network
        self.device = device
        self.num_workers = num_workers
        self.sims_per_worker = sims_per_worker
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature
        
        # Get model state for sharing with workers
        self.model_state = network.state_dict()
    
    def run(self, board: chess.Board, add_noise: bool = False) -> dict[chess.Move, int]:
        """
        Run parallel MCTS and return merged visit counts.
        
        Returns dict mapping moves to total visit counts across all workers.
        """
        fen = board.fen()
        device_name = str(self.device)
        
        # Prepare worker arguments
        worker_args = [
            (
                worker_id,
                fen,
                self.sims_per_worker,
                self.model_state,
                device_name,
                self.c_puct,
                add_noise,
                self.dirichlet_alpha,
                self.batch_size,
            )
            for worker_id in range(self.num_workers)
        ]
        
        # Run workers in parallel
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.starmap(_worker_search, worker_args)
        
        # Merge visit counts from all workers
        merged_counts: dict[str, int] = {}
        for worker_result in results:
            for move_uci, count in worker_result.items():
                merged_counts[move_uci] = merged_counts.get(move_uci, 0) + count
        
        # Convert back to Move objects
        move_counts = {}
        for move in board.legal_moves:
            uci = move.uci()
            if uci in merged_counts:
                move_counts[move] = merged_counts[uci]
        
        return move_counts
    
    def get_policy(self, move_counts: dict[chess.Move, int]) -> tuple[list[chess.Move], np.ndarray]:
        """Convert visit counts to move probabilities."""
        moves = list(move_counts.keys())
        counts = np.array([move_counts[m] for m in moves], dtype=np.float64)
        
        if self.temperature == 0:
            policy = np.zeros(len(moves))
            policy[counts.argmax()] = 1.0
        else:
            counts_t = counts ** (1.0 / self.temperature)
            policy = counts_t / counts_t.sum()
        
        return moves, policy
    
    def best_move(self, board: chess.Board, add_noise: bool = False) -> chess.Move:
        """Run parallel MCTS and return the best move."""
        move_counts = self.run(board, add_noise=add_noise)
        moves, probs = self.get_policy(move_counts)
        return moves[int(probs.argmax())]


if __name__ == "__main__":
    import time
    
    # Must use spawn for CUDA
    mp.set_start_method('spawn', force=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    net = ChessNet(num_res_blocks=4, channels=128).to(device)
    
    board = chess.Board()
    
    print("\n=== Multiprocess MCTS (4 workers × 200 sims) ===")
    mcts_mp = MultiprocessMCTS(
        network=net,
        device=device,
        num_workers=4,
        sims_per_worker=200,
        batch_size=16,
        c_puct=2.5,
    )
    
    start = time.time()
    move_counts = mcts_mp.run(board, add_noise=False)
    mp_time = time.time() - start
    
    total_sims = sum(move_counts.values())
    print(f"Time: {mp_time:.2f}s ({total_sims/mp_time:.1f} sims/sec)")
    print(f"Total simulations: {total_sims}")
    
    # Show top moves
    moves, probs = mcts_mp.get_policy(move_counts)
    print(f"\nTop 5 moves:")
    for m, p in sorted(zip(moves, probs), key=lambda x: -x[1])[:5]:
        visits = move_counts[m]
        print(f"  {board.san(m):6s}  prob={p:.3f}  visits={visits}")
