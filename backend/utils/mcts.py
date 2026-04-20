"""
mcts_batched.py  (optimised)
----------------------------
Key changes vs v1:
  • Board tensor cached on each MCTSNode — computed once, reused forever.
  • Legal-move mask cached on each MCTSNode — same reason.
  • Pre-allocated GPU batch buffer — no torch.stack() allocation every call.
  • torch.inference_mode instead of no_grad (slightly faster).
  • _select now returns a flat path list, avoiding repeated list appends.
  • Removed redundant board.copy() calls during expansion.
"""

from __future__ import annotations

import math
import chess
import numpy as np
import torch
import torch.nn.functional as F

from board_encoder import (
    board_to_tensor, legal_moves_mask,
    move_to_action, NUM_ACTIONS, NUM_PLANES, BOARD_SIZE,
)
from network import ChessNet


# ---------------------------------------------------------------------------
# MCTS Node  (tensor cache added)
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = (
        "board", "parent", "move", "children",
        "prior", "visit_count", "value_sum",
        "virtual_loss", "_is_expanded",
        "_tensor", "_mask",           # ← cached GPU-ready tensors
    )

    def __init__(
        self,
        board: chess.Board,
        parent: "MCTSNode | None" = None,
        move: "chess.Move | None" = None,
        prior: float = 0.0,
    ):
        self.board        = board
        self.parent       = parent
        self.move         = move
        self.children:    dict[chess.Move, MCTSNode] = {}
        self.prior        = prior
        self.visit_count  = 0
        self.value_sum    = 0.0
        self.virtual_loss = 0
        self._is_expanded = False
        self._tensor      = None   # set on first access
        self._mask        = None   # set on first access

    def get_tensor(self, device: torch.device) -> torch.Tensor:
        """(18,8,8) board tensor — computed once and cached on CPU."""
        if self._tensor is None:
            self._tensor = board_to_tensor(self.board)   # stays on CPU
        return self._tensor.to(device, non_blocking=True)

    def get_mask(self, device: torch.device) -> torch.Tensor:
        """(NUM_ACTIONS,) legal-move mask — computed once and cached on CPU."""
        if self._mask is None:
            self._mask = legal_moves_mask(self.board)    # stays on CPU
        return self._mask.to(device, non_blocking=True)

    @property
    def q_value(self) -> float:
        n = self.visit_count + self.virtual_loss
        if n == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / n

    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        n = self.visit_count + self.virtual_loss
        u = c_puct * self.prior * math.sqrt(max(parent_visits, 1)) / (1 + n)
        return self.q_value + u

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def terminal_value(self) -> float:
        outcome = self.board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner != self.board.turn else -1.0


# ---------------------------------------------------------------------------
# Batched MCTS
# ---------------------------------------------------------------------------

class BatchedMCTS:
    """
    MCTS with batched GPU leaf evaluation and cached board tensors.

    Args:
        network      : ChessNet
        device       : torch device
        num_sims     : total simulations per move
        num_parallel : leaves per GPU batch — set equal to num_sims for
                       exactly ONE forward pass per move (fastest)
        c_puct       : exploration constant
        dirichlet_alpha  : Dirichlet noise alpha for root
        dirichlet_weight : fraction of prior replaced by noise
        temperature  : for move sampling
    """

    def __init__(
        self,
        network: ChessNet,
        device: torch.device | None = None,
        num_sims: int = 200,
        num_parallel: int = 64,
        c_puct: float = 2.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        temperature: float = 1.0,
    ):
        self.net              = network
        self.device           = device or torch.device("cpu")
        self.num_sims         = num_sims
        self.num_parallel     = num_parallel
        self.c_puct           = c_puct
        self.dirichlet_alpha  = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.temperature      = temperature

        # Pre-allocate reusable CPU batch buffer to avoid repeated allocations
        self._buf = torch.zeros(num_parallel, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

    # ------------------------------------------------------------------
    # Selection (with virtual loss)
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode) -> tuple[MCTSNode, list[MCTSNode]]:
        node = root
        path = [node]
        node.visit_count  += 1
        node.virtual_loss += 1

        while node._is_expanded and not node.is_terminal():
            pv = node.visit_count + node.virtual_loss
            node = max(
                node.children.values(),
                key=lambda c: c.puct_score(self.c_puct, pv),
            )
            node.visit_count  += 1
            node.virtual_loss += 1
            path.append(node)

        return node, path

    def _undo_virtual_loss(self, path: list[MCTSNode]) -> None:
        for n in path:
            n.virtual_loss -= 1
            n.visit_count  -= 1   # will be re-incremented by backprop

    # ------------------------------------------------------------------
    # Batched expand + evaluate
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _batch_evaluate(self, leaves: list[MCTSNode]) -> list[float]:
        """Evaluate all leaves in a single GPU forward pass."""
        values           = [None] * len(leaves)
        non_terminal_idx = []

        for i, leaf in enumerate(leaves):
            if leaf.is_terminal():
                values[i] = leaf.terminal_value()
            else:
                non_terminal_idx.append(i)

        if not non_terminal_idx:
            return values

        B = len(non_terminal_idx)

        # Fill pre-allocated buffer (avoids torch.stack allocation)
        if B <= self.num_parallel:
            buf = self._buf[:B]
        else:
            buf = torch.zeros(B, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

        for slot, i in enumerate(non_terminal_idx):
            buf[slot].copy_(leaves[i].get_tensor(torch.device("cpu")))

        batch_tensors = buf.to(self.device, non_blocking=True)

        # Stack masks for illegal-move filtering
        batch_masks = torch.stack([
            leaves[i].get_mask(self.device) for i in non_terminal_idx
        ])  # (B, NUM_ACTIONS)

        # ── ONE GPU CALL ──────────────────────────────────────────────
        self.net.eval()
        policy_logits, value_preds = self.net(batch_tensors)

        policy_logits = policy_logits.masked_fill(~batch_masks, float("-inf"))
        policy_probs  = F.softmax(policy_logits, dim=1).cpu().numpy()  # (B, 4672)
        value_preds   = value_preds.cpu().numpy()                       # (B, 1)

        # Expand each non-terminal leaf
        for slot, leaf_idx in enumerate(non_terminal_idx):
            leaf     = leaves[leaf_idx]
            probs_np = policy_probs[slot]
            val      = float(value_preds[slot, 0])

            for move in leaf.board.legal_moves:
                child_board = leaf.board.copy(stack=False)
                child_board.push(move)
                child = MCTSNode(
                    board=child_board,
                    parent=leaf,
                    move=move,
                    prior=float(probs_np[move_to_action(move)]),
                )
                leaf.children[move] = child

            leaf._is_expanded = True
            values[leaf_idx]  = -val   # flip: value is from parent's POV

        return values

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum   += value
            value             = -value

    # ------------------------------------------------------------------
    # Dirichlet noise
    # ------------------------------------------------------------------

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        moves = list(root.children.keys())
        if not moves:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        w     = self.dirichlet_weight
        for move, n in zip(moves, noise):
            root.children[move].prior = (1 - w) * root.children[move].prior + w * n

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, board: chess.Board, add_noise: bool = False) -> MCTSNode:
        """
        Run simulations from `board`.

        With num_parallel == num_sims: exactly ONE GPU call per move.
        With num_parallel <  num_sims: ceil(num_sims/num_parallel) GPU calls.
        """
        root = MCTSNode(board=board.copy(stack=False))

        # Initial expansion (populates root.children with priors)
        self._batch_evaluate([root])
        root.visit_count  = 0
        root.value_sum    = 0.0
        root.virtual_loss = 0

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        num_batches = math.ceil(self.num_sims / self.num_parallel)

        for _ in range(num_batches):
            leaves = []
            paths  = []

            for _ in range(self.num_parallel):
                leaf, path = self._select(root)
                leaves.append(leaf)
                paths.append(path)

            values = self._batch_evaluate(leaves)

            for path, value in zip(paths, values):
                self._undo_virtual_loss(path)
                self._backpropagate(path, value if value is not None else 0.0)

        return root

    def get_policy(
        self, root: MCTSNode
    ) -> tuple[list[chess.Move], np.ndarray]:
        moves  = list(root.children.keys())
        counts = np.array(
            [root.children[m].visit_count for m in moves], dtype=np.float64
        )

        if self.temperature == 0 or counts.sum() == 0:
            best   = int(np.argmax(counts))
            policy = np.zeros(len(moves))
            policy[best] = 1.0
        else:
            ct     = counts ** (1.0 / self.temperature)
            policy = ct / ct.sum()

        return moves, policy

    def best_move(self, board: chess.Board, add_noise: bool = False) -> chess.Move:
        root         = self.run(board, add_noise=add_noise)
        moves, probs = self.get_policy(root)
        return moves[int(np.argmax(probs))]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import sys
    sys.path.insert(0, ".")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    from network import ChessNet
    net   = ChessNet(num_res_blocks=4, channels=128).to(device)
    board = chess.Board()
    N     = 50   # sims per benchmark run

    print(f"Benchmarking {N} sims, various num_parallel values:\n")
    print(f"{'num_parallel':>14}  {'time':>8}  {'moves/s':>10}  {'GPU calls/move':>16}")
    print("-" * 56)

    for np_val in [1, 8, 16, 32, N]:
        mcts = BatchedMCTS(net, device=device, num_sims=N, num_parallel=np_val)
        # warmup
        mcts.run(board)
        # timed
        t0 = time.perf_counter()
        REPS = 5
        for _ in range(REPS):
            mcts.run(board)
        elapsed   = (time.perf_counter() - t0) / REPS
        import math
        gpu_calls = math.ceil(N / np_val)
        print(f"{np_val:>14}  {elapsed:>7.3f}s  {1/elapsed:>10.1f}  {gpu_calls:>16}")