"""
mcts_simple.py
--------------
Barebones MCTS implementation for chess - prioritizes readability over speed.

Removed optimizations:
  - No batched GPU evaluation (evaluates one leaf at a time)
  - No tensor caching (recomputes on every access)
  - No virtual loss (simple serial tree traversal)
  - No pre-allocated buffers
"""

from __future__ import annotations

import math
import chess
import numpy as np
import torch

from board_encoder import board_to_tensor, legal_moves_mask, move_to_action, NUM_ACTIONS
from network import ChessNet

class MCTSNode:
    """A node in the search tree."""
    
    def __init__(
        self,
        board: chess.Board,
        parent: MCTSNode | None = None,
        move: chess.Move | None = None,
        prior: float = 0.0,
    ):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: dict[chess.Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    @property
    def q_value(self) -> float:
        """Average value from all visits."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        """PUCT formula: Q(s,a) + U(s,a)"""
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u


class MCTS:
    """
    Simple serial MCTS - one simulation at a time.
    
    Args:
        network: ChessNet policy-value network
        device: torch device
        num_sims: number of simulations per move
        c_puct: exploration constant
        dirichlet_alpha: Dirichlet noise parameter
        temperature: move sampling temperature
    """
    
    def __init__(
        self,
        network: ChessNet,
        device: torch.device,
        num_sims: int = 100,
        c_puct: float = 2.5,
        dirichlet_alpha: float = 0.3,
        temperature: float = 1.0,
    ):
        self.net = network
        self.device = device
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature
    
    def run(self, board: chess.Board, add_noise: bool = False) -> MCTSNode:
        """Run num_sims simulations from the given board position."""
        root = MCTSNode(board=board.copy(stack=False))
        
        # Expand root first
        self._expand(root)
        
        # Add Dirichlet noise to root children (for exploration during training)
        if add_noise and root.children:
            self._add_noise(root)
        
        # Run simulations
        for _ in range(self.num_sims):
            node = root
            path = [node]
            
            # Selection: traverse tree using PUCT until we hit a leaf
            while node.is_expanded and not node.board.is_game_over():
                node = max(
                    node.children.values(),
                    key=lambda c: c.puct_score(self.c_puct, node.visit_count)
                )
                path.append(node)
            
            # Expansion and evaluation
            if node.board.is_game_over():
                # Terminal node: use game result
                value = self._terminal_value(node)
            else:
                # Leaf node: expand and evaluate with network
                self._expand(node)
                value = self._evaluate(node)
            
            # Backpropagation: update all nodes on the path
            self._backprop(path, value)
        
        return root
    
    @torch.no_grad()
    def _evaluate(self, node: MCTSNode) -> float:
        """Evaluate leaf node using the network."""
        tensor = board_to_tensor(node.board, device=self.device).unsqueeze(0)  # (1,18,8,8)
        mask = legal_moves_mask(node.board, device=self.device).unsqueeze(0)   # (1,4672)
        
        self.net.eval()
        policy_logits, value = self.net(tensor)
        
        # Value is from perspective of player to move at this node
        # Since we just moved INTO this node, flip the sign
        return -float(value[0, 0])
    
    @torch.no_grad()
    def _expand(self, node: MCTSNode) -> None:
        """Expand node by creating children with network priors."""
        tensor = board_to_tensor(node.board, device=self.device).unsqueeze(0)
        mask = legal_moves_mask(node.board, device=self.device).unsqueeze(0)
        
        self.net.eval()
        policy_logits, _ = self.net(tensor)
        policy_logits = policy_logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
        
        # Create child nodes
        for move in node.board.legal_moves:
            child_board = node.board.copy(stack=False)
            child_board.push(move)
            prior = float(probs[move_to_action(move)])
            node.children[move] = MCTSNode(
                board=child_board,
                parent=node,
                move=move,
                prior=prior,
            )
        
        node.is_expanded = True
    
    def _terminal_value(self, node: MCTSNode) -> float:
        """Return game outcome from current player's perspective."""
        outcome = node.board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        # Winner is the side that just moved (previous player)
        return 1.0 if outcome.winner != node.board.turn else -1.0
    
    def _backprop(self, path: list[MCTSNode], value: float) -> None:
        """Update visit counts and value sums along path."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # flip sign at each level
    
    def _add_noise(self, root: MCTSNode) -> None:
        """Add Dirichlet noise to root priors for exploration."""
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = 0.75 * child.prior + 0.25 * n
    
    def get_policy(self, root: MCTSNode) -> tuple[list[chess.Move], np.ndarray]:
        """Convert visit counts to move probabilities."""
        moves = list(root.children.keys())
        counts = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
        
        if self.temperature == 0:
            # Deterministic: pick most visited
            policy = np.zeros(len(moves))
            policy[counts.argmax()] = 1.0
        else:
            # Stochastic: sample proportional to visits^(1/temp)
            counts_t = counts ** (1.0 / self.temperature)
            policy = counts_t / counts_t.sum()
        
        return moves, policy
    
    def best_move(self, board: chess.Board, add_noise: bool = False) -> chess.Move:
        """Run MCTS and return the best move."""
        root = self.run(board, add_noise=add_noise)
        moves, probs = self.get_policy(root)
        return moves[int(probs.argmax())]


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    net = ChessNet(num_res_blocks=4, channels=128).to(device)
    mcts = MCTS(net, device, num_sims=50)
    
    board = chess.Board()
    root = mcts.run(board, add_noise=True)
    moves, probs = mcts.get_policy(root)
    
    print(f"\nTop 3 moves from starting position:")
    for m, p in sorted(zip(moves, probs), key=lambda x: -x[1])[:3]:
        print(f"  {board.san(m):6s}  prob={p:.3f}")