"""
train.py
--------
Self-play training loop for the chess policy-value network.

Speed guide (num_sims is the dominant knob):
  num_sims=  25, num_parallel=16  →  fast debugging run
  num_sims= 100, num_parallel=64  →  reasonable training quality
  num_sims= 400, num_parallel=128 →  AlphaZero-style (slow)
"""

from __future__ import annotations

import platform
import random
import time
import collections
from dataclasses import dataclass
from typing import NamedTuple

import chess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from board_encoder import board_to_tensor, legal_moves_mask, move_to_action, NUM_ACTIONS
from network import ChessNet, AlphaZeroLoss, count_parameters
from mcts import BatchedMCTS as MCTS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Network
    num_res_blocks : int   = 4
    channels       : int   = 128

    # MCTS
    # Set num_parallel == num_sims for exactly ONE GPU call per move (fastest).
    # Increase num_sims for stronger play; keep num_parallel <= num_sims.
    num_sims       : int   = 50     # total sims per move
    num_parallel   : int   = 50     # boards per GPU batch (= num_sims → 1 call/move) (GPU saturation knob)
    c_puct         : float = 2.5
    dirichlet_alpha: float = 0.3

    # Self-play
    games_per_iter : int   = 10
    max_game_moves : int   = 200

    # Training
    replay_buffer_size : int   = 20_000
    batch_size         : int   = 128
    train_steps        : int   = 200
    lr                 : float = 1e-3
    weight_decay       : float = 1e-4
    num_iterations     : int   = 50

    # Misc
    device           : str  = "cuda"
    seed             : int  = 42
    checkpoint_every : int  = 5
    # torch.compile requires Triton, which is Linux-only.
    # We auto-disable it on Windows; you can force it with use_compile=True.
    use_compile      : bool = False


cfg = Config()


def _try_compile(net: ChessNet) -> ChessNet:
    """Enable torch.compile only when Triton is available (Linux + CUDA)."""
    if not cfg.use_compile:
        print("torch.compile : disabled (use_compile=False)")
        return net
    if platform.system() == "Windows":
        print("torch.compile : disabled (Triton not supported on Windows)")
        return net
    if not hasattr(torch, "compile"):
        print("torch.compile : disabled (PyTorch < 2.0)")
        return net
    try:
        compiled = torch.compile(net)
        # Trigger a test compile now so we fail fast rather than mid-training
        dummy = torch.zeros(1, 18, 8, 8, device=next(net.parameters()).device)
        compiled(dummy)
        print("torch.compile : enabled ✓")
        return compiled
    except Exception as e:
        print(f"torch.compile : disabled ({e})")
        return net


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class GameSample(NamedTuple):
    board_tensor : torch.Tensor   # (18, 8, 8) on CPU
    mcts_policy  : torch.Tensor   # (NUM_ACTIONS,)
    outcome      : float          # +1 / 0 / -1 from current player's POV


class ReplayBuffer:
    def __init__(self, max_size: int):
        self._buf: collections.deque[GameSample] = collections.deque(maxlen=max_size)

    def add(self, samples: list[GameSample]):
        self._buf.extend(samples)

    def sample(self, n: int) -> list[GameSample]:
        return random.sample(self._buf, min(n, len(self._buf)))

    def __len__(self):
        return len(self._buf)


class GameSampleDataset(Dataset):
    def __init__(self, samples: list[GameSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s.board_tensor,
            s.mcts_policy,
            torch.tensor([s.outcome], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Self-play — one game
# ---------------------------------------------------------------------------

def self_play_game(
    net: ChessNet,
    mcts_engine: MCTS,
    device: torch.device,
    max_moves: int,
    temp_threshold: int = 30,
    move_bar: tqdm | None = None,
) -> list[GameSample]:
    """
    Play one self-play game.  `move_bar` is a tqdm bar that tracks moves
    across all games in the iteration — pass it in from the caller so it
    updates in real time.
    """
    board    = chess.Board()
    history  : list[tuple[chess.Board, torch.Tensor]] = []
    move_num = 0

    while not board.is_game_over() and move_num < max_moves:
        mcts_engine.temperature = 1.0 if move_num < temp_threshold else 0.0

        root         = mcts_engine.run(board, add_noise=True)
        moves, probs = mcts_engine.get_policy(root)

        policy_vec = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        for m, p in zip(moves, probs):
            policy_vec[move_to_action(m)] = p

        history.append((board.copy(stack=False), policy_vec))

        chosen_idx = (
            np.random.choice(len(moves), p=probs)
            if mcts_engine.temperature > 0
            else int(np.argmax(probs))
        )
        board.push(moves[chosen_idx])
        move_num += 1

        if move_bar is not None:
            move_bar.update(1)

    outcome     = board.outcome()
    draw        = outcome is None or outcome.winner is None
    white_score = 0.0 if draw else (1.0 if outcome.winner == chess.WHITE else -1.0)
    result_str  = "½-½" if draw else ("1-0" if white_score > 0 else "0-1")

    if move_bar is not None:
        move_bar.set_postfix_str(f"last result: {result_str} ({move_num} plies)")

    samples = []
    for ply_board, policy_vec in history:
        z = 0.0 if draw else (white_score if ply_board.turn == chess.WHITE else -white_score)
        tensor = board_to_tensor(ply_board, device=device)
        samples.append(GameSample(tensor.cpu(), policy_vec, z))

    return samples, result_str, move_num


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
    totals    = {"loss": 0.0, "policy": 0.0, "value": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc="  training", unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for board_tensors, policy_targets, value_targets in pbar:
        board_tensors  = board_tensors.to(device, non_blocking=True)
        policy_targets = policy_targets.to(device, non_blocking=True)
        value_targets  = value_targets.to(device, non_blocking=True)

        policy_logits, value_pred = net(board_tensors)
        loss, p_loss, v_loss = criterion(
            policy_logits, value_pred, policy_targets, value_targets
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        totals["loss"]   += loss.item()
        totals["policy"] += p_loss.item()
        totals["value"]  += v_loss.item()
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", p=f"{p_loss.item():.4f}", v=f"{v_loss.item():.4f}")

    d = max(n_batches, 1)
    return {k: v / d for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # Enable TF32 for a free ~10% speedup on Ampere+ GPUs (RTX 30xx/40xx)
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Training on : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    net = ChessNet(num_res_blocks=cfg.num_res_blocks, channels=cfg.channels).to(device)
    print(f"Parameters  : {count_parameters(net):,}")
    net = _try_compile(net)

    optimizer   = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_iterations, eta_min=1e-5)
    criterion   = AlphaZeroLoss()
    buffer      = ReplayBuffer(cfg.replay_buffer_size)
    mcts_engine = MCTS(
        network=net,
        device=device,
        num_sims=cfg.num_sims,
        num_parallel=cfg.num_parallel,
        c_puct=cfg.c_puct,
        dirichlet_alpha=cfg.dirichlet_alpha,
    )

    print(
        f"\nConfig: {cfg.num_sims} sims/move  |  batch={cfg.num_parallel}  |  "
        f"{cfg.games_per_iter} games/iter  |  {cfg.num_res_blocks} res-blocks  |  {cfg.channels} ch\n"
    )

    for iteration in range(1, cfg.num_iterations + 1):
        iter_t0 = time.time()
        print(f"{'='*60}")
        print(f"  Iteration {iteration}/{cfg.num_iterations}")
        print(f"{'='*60}")

        # ── Self-play ────────────────────────────────────────────────────────
        new_samples : list[GameSample] = []
        results     : list[str]        = []

        # One tqdm bar counting moves across ALL games in this iteration
        move_bar = tqdm(
            total=cfg.games_per_iter * cfg.max_game_moves,
            desc=f"  self-play (0/{cfg.games_per_iter} games)",
            unit="move",
            bar_format="{l_bar}{bar}| {n_fmt} moves [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )

        for g in range(cfg.games_per_iter):
            move_bar.set_description(f"  self-play ({g+1}/{cfg.games_per_iter} games)")
            samples, result, n_plies = self_play_game(
                net, mcts_engine, device,
                max_moves=cfg.max_game_moves,
                move_bar=move_bar,
            )
            new_samples.extend(samples)
            results.append(result)

        move_bar.close()

        result_summary = f"  W:{results.count('1-0')}  D:{results.count('½-½')}  L:{results.count('0-1')}"
        buffer.add(new_samples)
        sp_time = time.time() - iter_t0
        print(f"{result_summary}  |  {len(new_samples)} positions  |  buffer={len(buffer):,}  |  {sp_time:.1f}s\n")

        # ── Training ─────────────────────────────────────────────────────────
        if len(buffer) < cfg.batch_size:
            print("  Buffer too small — skipping training.\n")
            continue

        train_t0 = time.time()
        samples  = buffer.sample(cfg.train_steps * cfg.batch_size)
        dataset  = GameSampleDataset(samples)
        loader   = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        metrics = train_epoch(net, optimizer, criterion, loader, device)
        scheduler.step()

        print(
            f"\n  loss={metrics['loss']:.4f}  "
            f"policy={metrics['policy']:.4f}  "
            f"value={metrics['value']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({time.time()-train_t0:.1f}s)"
        )
        print(f"  Iteration total: {time.time()-iter_t0:.1f}s\n")

        if iteration % cfg.checkpoint_every == 0:
            raw_net = getattr(net, "_orig_mod", net)
            path = f"checkpoint_iter{iteration:04d}.pt"
            torch.save({
                "iteration" : iteration,
                "config"    : cfg,
                "model"     : raw_net.state_dict(),
                "optimizer" : optimizer.state_dict(),
            }, path)
            print(f"  ✓ Checkpoint → {path}\n")

    print("Training complete.")


if __name__ == "__main__":
    main()