"""
train.py
--------
Self-play training loop for the chess policy-value network.

Pipeline (AlphaZero-style):
  1. Self-play   : play games using MCTS + current network, collecting
                   (board_tensor, mcts_policy, outcome) triplets.
  2. Train       : sample mini-batches from the replay buffer and update
                   the network using the AlphaZeroLoss.
  3. Evaluate    : (optional) pit new network vs old; keep winner.
  4. Repeat.

Usage:
    python train.py

Adjust the Config dataclass below to change hyperparameters.
"""

from __future__ import annotations

import random
import collections
from dataclasses import dataclass, field
from typing import NamedTuple

import chess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from board_encoder import board_to_tensor, legal_moves_mask, move_to_action, NUM_ACTIONS
from network import ChessNet, AlphaZeroLoss, count_parameters
from mcts import MCTS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Network
    num_res_blocks : int   = 10
    channels       : int   = 256

    # MCTS
    num_sims       : int   = 200
    c_puct         : float = 2.5
    dirichlet_alpha: float = 0.3

    # Self-play
    games_per_iter : int   = 100    # games per training iteration
    max_game_moves : int   = 512    # cap to avoid infinite games

    # Training
    replay_buffer_size : int   = 50_000
    batch_size         : int   = 256
    train_steps        : int   = 1000  # gradient steps per iteration
    lr                 : float = 1e-3
    weight_decay       : float = 1e-4
    num_iterations     : int   = 50

    # Misc
    device : str = "cuda"  # "cpu" or "cuda"
    seed   : int = 42
    checkpoint_every : int = 5   # save every N iterations


cfg = Config()


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class GameSample(NamedTuple):
    board_tensor   : torch.Tensor   # (18, 8, 8)
    mcts_policy    : torch.Tensor   # (NUM_ACTIONS,)
    outcome        : float          # +1 win / 0 draw / -1 loss (current player POV)


class ReplayBuffer:
    def __init__(self, max_size: int):
        self._buffer: collections.deque[GameSample] = collections.deque(maxlen=max_size)

    def add(self, samples: list[GameSample]):
        self._buffer.extend(samples)

    def sample(self, n: int) -> list[GameSample]:
        return random.sample(self._buffer, min(n, len(self._buffer)))

    def __len__(self):
        return len(self._buffer)


class GameSampleDataset(Dataset):
    def __init__(self, samples: list[GameSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s.board_tensor, s.mcts_policy, torch.tensor([s.outcome], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def self_play_game(
    net: ChessNet,
    mcts: MCTS,
    device: torch.device,
    max_moves: int = 512,
    temp_threshold: int = 30,  # use temperature=1 for first N plies, then 0
) -> list[GameSample]:
    """
    Play one game of self-play.  Returns a list of GameSamples, one per ply,
    with outcomes assigned retroactively from the game result.
    """
    board    = chess.Board()
    history  : list[tuple[chess.Board, torch.Tensor]] = []  # (board_copy, mcts_policy)

    move_num = 0

    while not board.is_game_over() and move_num < max_moves:
        # Temperature schedule: explore early, exploit late
        mcts.temperature = 1.0 if move_num < temp_threshold else 0.0

        root         = mcts.run(board, add_noise=True)
        moves, probs = mcts.get_policy(root)

        # Build full action-space policy vector
        policy_vec = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        for m, p in zip(moves, probs):
            policy_vec[move_to_action(m)] = p

        history.append((board.copy(stack=False), policy_vec))

        # Sample (or argmax) a move
        if mcts.temperature > 0:
            chosen_idx = np.random.choice(len(moves), p=probs)
        else:
            chosen_idx = int(np.argmax(probs))
        board.push(moves[chosen_idx])
        move_num += 1

    # Outcome
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        final_result = 0.0  # draw or max-moves reached
    else:
        # outcome.winner is the color that won
        final_result_white = 1.0 if outcome.winner == chess.WHITE else -1.0

    samples = []
    for ply_board, policy_vec in history:
        if outcome is None or outcome.winner is None:
            z = 0.0
        else:
            z = final_result_white if ply_board.turn == chess.WHITE else -final_result_white
        tensor = board_to_tensor(ply_board, device=device)
        samples.append(GameSample(tensor.cpu(), policy_vec, z))

    return samples


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_epoch(
    net: ChessNet,
    optimizer: optim.Optimizer,
    criterion: AlphaZeroLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    net.train()
    total_loss_sum   = 0.0
    policy_loss_sum  = 0.0
    value_loss_sum   = 0.0
    n_batches        = 0

    for board_tensors, policy_targets, value_targets in loader:
        board_tensors   = board_tensors.to(device)
        policy_targets  = policy_targets.to(device)
        value_targets   = value_targets.to(device)

        policy_logits, value_pred = net(board_tensors)
        loss, p_loss, v_loss      = criterion(
            policy_logits, value_pred, policy_targets, value_targets
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        total_loss_sum  += loss.item()
        policy_loss_sum += p_loss.item()
        value_loss_sum  += v_loss.item()
        n_batches       += 1

    return {
        "loss"        : total_loss_sum  / max(n_batches, 1),
        "policy_loss" : policy_loss_sum / max(n_batches, 1),
        "value_loss"  : value_loss_sum  / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Build network
    net = ChessNet(
        num_res_blocks=cfg.num_res_blocks,
        channels=cfg.channels,
    ).to(device)
    print(f"Parameters: {count_parameters(net):,}")

    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_iterations, eta_min=1e-5
    )
    criterion = AlphaZeroLoss()
    buffer    = ReplayBuffer(cfg.replay_buffer_size)

    mcts_engine = MCTS(
        network=net,
        device=device,
        num_sims=cfg.num_sims,
        c_puct=cfg.c_puct,
        dirichlet_alpha=cfg.dirichlet_alpha,
    )

    for iteration in range(1, cfg.num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{cfg.num_iterations}")
        print(f"{'='*60}")

        # ── Self-play phase ─────────────────────────────────────────
        print(f"  Self-play: {cfg.games_per_iter} games …")
        new_samples = []
        for g in range(cfg.games_per_iter):
            samples = self_play_game(
                net, mcts_engine, device, max_moves=cfg.max_game_moves
            )
            new_samples.extend(samples)
            if (g + 1) % 10 == 0:
                print(f"    {g+1}/{cfg.games_per_iter} games  ({len(new_samples)} positions)")
        buffer.add(new_samples)
        print(f"  Replay buffer size: {len(buffer)}")

        # ── Training phase ──────────────────────────────────────────
        if len(buffer) < cfg.batch_size:
            print("  Buffer too small, skipping training.")
            continue

        samples  = buffer.sample(cfg.train_steps * cfg.batch_size)
        dataset  = GameSampleDataset(samples)
        loader   = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        print(f"  Training for {cfg.train_steps} steps …")
        metrics = train_epoch(net, optimizer, criterion, loader, device)
        scheduler.step()

        print(f"  Loss: {metrics['loss']:.4f}  "
              f"(policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f})  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Checkpoint ──────────────────────────────────────────────
        if iteration % cfg.checkpoint_every == 0:
            path = f"checkpoint_iter{iteration:04d}.pt"
            torch.save({
                "iteration" : iteration,
                "model"     : net.state_dict(),
                "optimizer" : optimizer.state_dict(),
            }, path)
            print(f"  Checkpoint saved → {path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
