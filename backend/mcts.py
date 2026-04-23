"""
mcts_parallel.py
----------------
Parallelized MCTS with batched leaf evaluation for efficient GPU usage.

Key optimizations:
  - Batched network evaluation (process multiple leaves at once)
  - Virtual loss (prevents multiple simulations from exploring the same path)
  - Configurable batch size for GPU efficiency
"""

from __future__ import annotations

import math
import chess
import numpy as np
import torch
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
        self.virtual_loss = 0  # For parallel simulations
        self.is_expanded = False
    
    @property
    def q_value(self) -> float:
        """Average value including virtual loss."""
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        return self.value_sum / total_visits
    
    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        """PUCT formula: Q(s,a) + U(s,a)"""
        total_visits = self.visit_count + self.virtual_loss

        # (heuristic)
        if self.q_value > 0.8:
            c_puct /= 4
            
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + total_visits)
        return self.q_value + u


class MCTS:
    """
    Batched MCTS with virtual loss for efficient parallel search.
    
    Args:
        network: ChessNet policy-value network
        device: torch device
        num_sims: total number of simulations
        batch_size: number of leaves to evaluate per batch
        c_puct: exploration constant
        virtual_loss_weight: penalty for nodes being evaluated
        dirichlet_alpha: Dirichlet noise parameter
        temperature: move sampling temperature
    """
    
    def __init__(
        self,
        network: ChessNet,
        device: torch.device,
        num_sims: int = 800,
        batch_size: int = 16,
        c_puct: float = 2,
        virtual_loss_weight: float = 1.0,
        dirichlet_alpha: float = 0.3,
        temperature: float = 1.0,
    ):
        self.net = network
        self.device = device
        self.num_sims = num_sims
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.virtual_loss_weight = virtual_loss_weight
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature
    
    def run(self, board: chess.Board, add_noise: bool = False) -> MCTSNode:
        """Run num_sims simulations with batched evaluation."""
        root = MCTSNode(board=board.copy(stack=False))

        # Expand root first
        self._expand_batch([root])

        if add_noise and root.children:
            self._add_noise(root)

        num_batches = (self.num_sims + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            batch_leaves = []
            batch_paths = []
            batch_depths = []

            sims_this_batch = min(self.batch_size, self.num_sims - batch_idx * self.batch_size)

            for _ in range(sims_this_batch):
                node = root
                path = [node]

                while node.is_expanded and not node.board.is_game_over():
                    node = max(
                        node.children.values(),
                        key=lambda c: c.puct_score(self.c_puct, node.visit_count)
                    )
                    path.append(node)

                for n in path:
                    n.virtual_loss += self.virtual_loss_weight

                batch_leaves.append(node)
                batch_paths.append(path)
                batch_depths.append(len(path) - 1)

            values = self._evaluate_batch(batch_leaves, batch_depths)

            for path, value in zip(batch_paths, values):
                for n in path:
                    n.virtual_loss -= self.virtual_loss_weight

                self._backprop(path, value)

        return root
    
    @torch.no_grad()
    def _evaluate_batch(self, nodes: list[MCTSNode], depths: list[int]) -> list[float]:
        """Evaluate a batch of leaf nodes."""
        if not nodes:
            return []

        values = []
        non_terminal_nodes = []
        non_terminal_indices = []

        for i, (node, depth) in enumerate(zip(nodes, depths)):
            if node.board.is_game_over():
                values.append(self._terminal_value(node, depth))
            else:
                non_terminal_nodes.append(node)
                non_terminal_indices.append(i)
                values.append(0.0)

        if non_terminal_nodes:
            self._expand_batch(non_terminal_nodes)

            batch_tensors = []
            batch_masks = []

            for node in non_terminal_nodes:
                tensor = board_to_tensor(node.board, device=self.device)
                mask = legal_moves_mask(node.board, device=self.device)
                batch_tensors.append(tensor)
                batch_masks.append(mask)

            batch_tensors = torch.stack(batch_tensors, dim=0)
            batch_masks = torch.stack(batch_masks, dim=0)

            self.net.eval()
            policy_logits, batch_values = self.net(batch_tensors)

            network_values = -batch_values[:, 0].cpu().numpy()

            for idx, val in zip(non_terminal_indices, network_values):
                values[idx] = float(val)

        return values
    
    @torch.no_grad()
    def _expand_batch(self, nodes: list[MCTSNode]) -> None:
        """Expand a batch of nodes with network priors."""
        if not nodes:
            return
        
        # Filter out already expanded or terminal nodes
        to_expand = [n for n in nodes if not n.is_expanded and not n.board.is_game_over()]
        if not to_expand:
            return
        
        # Create batched tensors
        batch_tensors = []
        batch_masks = []
        
        for node in to_expand:
            tensor = board_to_tensor(node.board, device=self.device)
            mask = legal_moves_mask(node.board, device=self.device)
            batch_tensors.append(tensor)
            batch_masks.append(mask)
        
        batch_tensors = torch.stack(batch_tensors, dim=0)
        batch_masks = torch.stack(batch_masks, dim=0)
        
        # Single forward pass
        self.net.eval()
        policy_logits, _ = self.net(batch_tensors)
        policy_logits = policy_logits.masked_fill(~batch_masks, float("-inf"))
        probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
        
        # Create children for each node
        for node, node_probs in zip(to_expand, probs):
            for move in node.board.legal_moves:
                child_board = node.board.copy(stack=False)
                child_board.push(move)
                prior = float(node_probs[move_to_action(move)])

                # -------- (heuristic) Slight bias towards checks and captures in order to promote attacking behaviours when winning/explore them more
                if node.board.is_capture(move) or child_board.is_check():
                    prior *= 1.2

                node.children[move] = MCTSNode(
                    board=child_board,
                    parent=node,
                    move=move,
                    prior=prior,
                )
            node.is_expanded = True
    
    def _terminal_value(self, node: MCTSNode, ply_from_root: int) -> float:
        """Return game outcome from current player's perspective."""
        outcome = node.board.outcome()

        # print(f"terminal node reached {ply_from_root} deep")
        # print(node.board)

        if outcome is None or outcome.winner is None:
            return 0.0

        MATE_BASE = 100000.0

        # In a checkmated position, board.turn is the loser.
        if outcome.winner != node.board.turn:
            return MATE_BASE - ply_from_root
        else:
            return -MATE_BASE + ply_from_root
    
    def _backprop(self, path: list[MCTSNode], value: float) -> None:
        """Update visit counts and value sums along path."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
    
    def _add_noise(self, root: MCTSNode) -> None:
        """Add Dirichlet noise to root priors."""
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
            policy = np.zeros(len(moves))
            policy[counts.argmax()] = 1.0
        else:
            counts_t = counts ** (1.0 / self.temperature)
            policy = counts_t / counts_t.sum()
        
        return moves, policy
    
    def best_move(self, board: chess.Board, add_noise: bool = False) -> chess.Move:
        """Run MCTS and return the best move."""
        root = self.run(board, add_noise=add_noise)

        moves, probs = self.get_policy(root)
        return moves[int(probs.argmax())]


if __name__ == "__main__":
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    net = ChessNet(num_res_blocks=4, channels=128).to(device)
    
    board = chess.Board()
    
    print("\n=== Parallel MCTS (batched) ===")
    mcts_parallel = MCTS(net, device, num_sims=800, batch_size=16)
    
    start = time.time()
    root_parallel = mcts_parallel.run(board)
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s ({800/parallel_time:.1f} sims/sec)")
    
    # Show top moves from parallel version
    moves, probs = mcts_parallel.get_policy(root_parallel)
    print(f"\nTop 5 moves:")
    for m, p in sorted(zip(moves, probs), key=lambda x: -x[1])[:5]:
        print(f"  {board.san(m):6s}  prob={p:.3f}")
