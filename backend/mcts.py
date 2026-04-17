"""
mcts.py
-------
Monte Carlo Tree Search (PUCT variant, as in AlphaZero) using ChessNet for
both prior policy estimation and leaf-node value estimation.

Key differences from classical MCTS:
  • No random rollouts — leaf value comes directly from the value head (v).
  • Node selection uses PUCT (Polynomial Upper Confidence Trees):
        U(s,a) = c_puct · P(s,a) · √N(s) / (1 + n(s,a))
        Q(s,a) + U(s,a)
  • Dirichlet noise added to the root priors to encourage exploration
    (only during training / self-play).
"""

from __future__ import annotations

import math
import chess
import numpy as np
import torch

from board_encoder import board_to_tensor, legal_moves_mask, action_to_move, move_to_action
from network import ChessNet

# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    A single node in the search tree.

    Attributes:
        board        : position at this node (copy, not view)
        parent       : parent MCTSNode or None for root
        move         : the chess.Move that led to this node (None for root)
        children     : dict[chess.Move, MCTSNode]
        prior        : P(s, a) from the network policy
        visit_count  : N(s, a) — number of times this node was visited
        value_sum    : W(s, a) — cumulative value from backpropagation
    """

    __slots__ = (
        "board", "parent", "move", "children",
        "prior", "visit_count", "value_sum", "_is_expanded",
    )

    def __init__(
        self,
        board: chess.Board,
        parent: MCTSNode | None = None,
        move: chess.Move | None = None,
        prior: float = 0.0,
    ):
        self.board       = board
        self.parent      = parent
        self.move        = move
        self.children:   dict[chess.Move, MCTSNode] = {}
        self.prior       = prior
        self.visit_count = 0
        self.value_sum   = 0.0
        self._is_expanded = False

    # ------------------------------------------------------------------

    @property
    def q_value(self) -> float:
        """Mean action-value Q(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return self._is_expanded

    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        """PUCT score used for child selection."""
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def terminal_value(self) -> float:
        """
        Returns +1 if the CURRENT player has won, -1 if they lost, 0 for draw.
        (Value is always from the perspective of the player to move at this node.)
        """
        outcome = self.board.outcome()
        if outcome is None:
            return 0.0
        if outcome.winner is None:
            return 0.0  # draw
        # winner is a chess.Color; current turn means the player who JUST moved
        # loses (they wouldn't be in a terminal state if they had more moves).
        won = outcome.winner == (not self.board.turn)  # last mover wins
        return 1.0 if won else -1.0

    def __repr__(self) -> str:
        return (
            f"MCTSNode(move={self.move}, "
            f"N={self.visit_count}, Q={self.q_value:.3f}, P={self.prior:.3f})"
        )


# ---------------------------------------------------------------------------
# MCTS engine
# ---------------------------------------------------------------------------

class MCTS:
    """
    AlphaZero-style MCTS.

    Args:
        network    : trained (or initialised) ChessNet
        device     : torch device for network inference
        num_sims   : number of simulations (tree expansions) per move
        c_puct     : exploration constant (AlphaZero default ≈ 1.0–5.0)
        dirichlet_alpha  : α for Dirichlet noise at root (chess: 0.3)
        dirichlet_weight : fraction of root prior replaced by noise (0.25)
        temperature      : τ for visit-count sampling (1.0 = proportional,
                           0 = deterministic argmax)
    """

    def __init__(
        self,
        network: ChessNet,
        device: torch.device | None = None,
        num_sims: int = 400,
        c_puct: float = 2.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        temperature: float = 1.0,
    ):
        self.net              = network
        self.device           = device or torch.device("cpu")
        self.num_sims         = num_sims
        self.c_puct           = c_puct
        self.dirichlet_alpha  = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.temperature      = temperature

    # ------------------------------------------------------------------
    # Core MCTS steps
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse the tree using PUCT until we reach an unexpanded or terminal node."""
        while node.is_expanded and not node.is_terminal():
            node = max(
                node.children.values(),
                key=lambda c: c.puct_score(self.c_puct, node.visit_count),
            )
        return node

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand the node using network priors and return the value estimate.
        Returns the game-outcome value for terminal nodes.
        """
        if node.is_terminal():
            return node.terminal_value()

        board  = node.board
        tensor = board_to_tensor(board, device=self.device)
        mask   = legal_moves_mask(board, device=self.device)
        probs, value = self.net.predict(tensor, mask)
        probs_np = probs.cpu().numpy()

        # Create children with network priors
        for move in board.legal_moves:
            action_idx = move_to_action(move)
            prior      = float(probs_np[action_idx])
            child_board = board.copy(stack=False)
            child_board.push(move)
            node.children[move] = MCTSNode(
                board=child_board,
                parent=node,
                move=move,
                prior=prior,
            )

        node._is_expanded = True
        # Value is from the perspective of the player who just moved INTO this node.
        # Since board.turn is now the NEXT player, flip the sign.
        return -value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Walk back up to the root, flipping value sign at each ply
        (value is always from the perspective of the current player).
        """
        while node is not None:
            node.visit_count += 1
            node.value_sum   += value
            value             = -value
            node              = node.parent

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Inject Dirichlet noise into root children priors (used during training)."""
        moves  = list(root.children.keys())
        noise  = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        w      = self.dirichlet_weight
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - w) * child.prior + w * n

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        board: chess.Board,
        add_noise: bool = False,
    ) -> MCTSNode:
        """
        Run `num_sims` simulations from the given board position and return
        the populated root node.

        Args:
            board     : current chess position
            add_noise : add Dirichlet noise to root (True during self-play)
        """
        root = MCTSNode(board=board.copy(stack=False))

        # Expand root immediately so we can add noise to its children
        self._expand_and_evaluate(root)
        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_sims):
            leaf  = self._select(root)
            value = self._expand_and_evaluate(leaf)
            self._backpropagate(leaf, value)

        return root

    def get_policy(self, root: MCTSNode) -> tuple[list[chess.Move], np.ndarray]:
        """
        Derive the improved policy π from visit counts.

        Args:
            root : root node after running simulations

        Returns:
            moves  : list of chess.Move
            policy : normalised probability distribution over those moves
                     (shaped by temperature τ)
        """
        moves   = list(root.children.keys())
        counts  = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

        if self.temperature == 0:
            # Deterministic: one-hot on the most visited move
            best = int(np.argmax(counts))
            policy = np.zeros(len(moves), dtype=np.float64)
            policy[best] = 1.0
        else:
            counts_t = counts ** (1.0 / self.temperature)
            policy   = counts_t / counts_t.sum()

        return moves, policy

    def best_move(self, board: chess.Board, add_noise: bool = False) -> chess.Move:
        """
        Convenience wrapper: run MCTS and return the best move.

        Args:
            board     : current position
            add_noise : whether to add Dirichlet noise (set True for self-play)
        """
        root         = self.run(board, add_noise=add_noise)
        moves, probs = self.get_policy(root)
        return moves[int(np.argmax(probs))]

    def get_visit_counts(self, root: MCTSNode) -> dict[chess.Move, int]:
        """Return raw visit counts keyed by move (useful for debugging)."""
        return {m: c.visit_count for m, c in root.children.items()}


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from network import ChessNet, count_parameters

    net  = ChessNet(num_res_blocks=10, channels=256).to(device)
    mcts = MCTS(net, device=device, num_sims=50, temperature=1.0)

    board = chess.Board()
    print(f"Starting position:\n{board}\n")

    root         = mcts.run(board, add_noise=True)
    moves, probs = mcts.get_policy(root)

    print("Top-5 moves by visit share:")
    top_k = sorted(zip(moves, probs), key=lambda x: -x[1])[:5]
    for m, p in top_k:
        n = root.children[m].visit_count
        q = root.children[m].q_value
        print(f"  {board.san(m):8s}  visits={n:4d}  prob={p:.3f}  Q={q:+.4f}")

    best = mcts.best_move(board)
    print(f"\nBest move: {board.san(best)}")
