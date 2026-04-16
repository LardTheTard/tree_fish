"""
Basic Monte Carlo Tree Search (MCTS) with PUCT exploration.

PUCT score: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

  Q(s,a)  - mean value of action a from state s
  P(s,a)  - prior probability (from a policy network, or uniform)
  N(s)    - visit count of parent node
  N(s,a)  - visit count of this edge
  c_puct  - exploration constant
"""

import math
import random
from collections import defaultdict


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, state, parent=None, prior=1.0):
        self.state   = state        # opaque game state
        self.parent  = parent
        self.prior   = prior        # P(s,a) that led here

        self.children: dict = {}    # action -> Node
        self.visit_count  = 0
        self.value_sum    = 0.0

    # ------------------------------------------------------------------
    @property
    def q_value(self):
        """Mean value from this node's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def puct_score(self, c_puct: float) -> float:
        """PUCT score used by the parent to select this child."""
        parent_visits = self.parent.visit_count if self.parent else 1
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    """
    Vanilla MCTS with PUCT selection.

    Requires a `game` object that implements:
        game.get_actions(state)          -> list of legal actions
        game.apply_action(state, action) -> new_state
        game.is_terminal(state)          -> bool
        game.get_result(state)           -> float  (from current player's POV, +1 win / -1 loss)
        game.get_prior(state, actions)   -> dict {action: prior_prob}  (uniform ok)
    """

    def __init__(self, game, c_puct: float = 1.4, n_simulations: int = 200):
        self.game         = game
        self.c_puct       = c_puct
        self.n_simulations = n_simulations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, root_state) -> dict:
        """Run simulations from root_state and return visit-count policy."""
        root = Node(root_state)
        self._expand(root)

        for _ in range(self.n_simulations):
            node  = self._select(root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        # Policy: normalised visit counts
        total = sum(c.visit_count for c in root.children.values())
        policy = {
            action: child.visit_count / total
            for action, child in root.children.items()
        }
        return policy

    def best_action(self, root_state):
        policy = self.search(root_state)
        return max(policy, key=policy.get)

    # ------------------------------------------------------------------
    # Four MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: Node) -> Node:
        """Descend the tree, always picking the highest-PUCT child."""
        while not node.is_leaf():
            node = max(
                node.children.values(),
                key=lambda c: c.puct_score(self.c_puct)
            )
        return node

    def _expand(self, node: Node):
        """Add all legal children to a leaf node."""
        if self.game.is_terminal(node.state):
            return
        actions = self.game.get_actions(node.state)
        priors  = self.game.get_prior(node.state, actions)
        for action in actions:
            child_state = self.game.apply_action(node.state, action)
            node.children[action] = Node(
                state  = child_state,
                parent = node,
                prior  = priors.get(action, 1.0 / len(actions)),
            )

    def _evaluate(self, node: Node) -> float:
        """
        Expand then rollout (random playout) to get a value estimate.
        In AlphaZero you'd replace the rollout with a value network call.
        """
        if self.game.is_terminal(node.state):
            return self.game.get_result(node.state)

        self._expand(node)          # expand before rollout
        return self._rollout(node.state)

    def _rollout(self, state) -> float:
        """Random playout until terminal; return result."""
        while not self.game.is_terminal(state):
            actions = self.game.get_actions(state)
            state   = self.game.apply_action(state, random.choice(actions))
        return self.game.get_result(state)

    def _backpropagate(self, node: Node, value: float):
        """
        Walk back to root, flipping value sign at each ply
        (assumes alternating two-player zero-sum game).
        """
        while node is not None:
            node.visit_count += 1
            node.value_sum   += value
            value = -value          # opponent's perspective
            node  = node.parent


# ---------------------------------------------------------------------------
# Demo game: Tic-Tac-Toe
# ---------------------------------------------------------------------------

class TicTacToe:
    """Minimal Tic-Tac-Toe for testing MCTS."""

    def get_actions(self, state):
        board, _ = state
        return [i for i, v in enumerate(board) if v == 0]

    def apply_action(self, state, action):
        board, player = state
        board = list(board)
        board[action] = player
        return (tuple(board), -player)

    def is_terminal(self, state):
        board, _ = state
        return self._winner(board) != 0 or all(v != 0 for v in board)

    def get_result(self, state):
        """Return +1 if the player who just moved won, -1 if they lost, 0 draw."""
        board, player = state
        winner = self._winner(board)
        # 'player' is the NEXT player to move, so the one who just moved is -player
        if winner == -player:
            return 1.0
        if winner == player:
            return -1.0
        return 0.0

    def get_prior(self, state, actions):
        """Uniform prior — plug in a policy network here."""
        p = 1.0 / len(actions)
        return {a: p for a in actions}

    def _winner(self, board):
        lines = [
            (0,1,2),(3,4,5),(6,7,8),   # rows
            (0,3,6),(1,4,7),(2,5,8),   # cols
            (0,4,8),(2,4,6),           # diagonals
        ]
        for a,b,c in lines:
            if board[a] == board[b] == board[c] != 0:
                return board[a]
        return 0

    def print_board(self, state):
        board, player = state
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in range(3):
            print(' '.join(symbols[board[row*3+col]] for col in range(3)))
        print(f"Next player: {'X' if player == 1 else 'O'}\n")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    game  = TicTacToe()
    mcts  = MCTS(game, c_puct=1.4, n_simulations=400)

    state = (tuple([0]*9), 1)   # empty board, X moves first

    print("=== MCTS Tic-Tac-Toe self-play ===\n")
    while not game.is_terminal(state):
        game.print_board(state)
        policy = mcts.search(state)
        action = max(policy, key=policy.get)
        print(f"Chosen action: {action}  (policy: { {a: f'{p:.2f}' for a,p in sorted(policy.items())} })\n")
        state  = game.apply_action(state, action)

    game.print_board(state)
    winner = game._winner(state[0])
    if winner == 1:
        print("X wins!")
    elif winner == -1:
        print("O wins!")
    else:
        print("Draw!")