"""
train_simple.py
---------------
Barebones self-play training loop - prioritizes readability over speed.

Removed optimizations:
  - No tqdm progress bars (simple print statements)
  - No torch.compile handling
  - No detailed timing/statistics
  - Uses simple serial MCTS (slower but clearer)
  - Minimal checkpoint format
"""

from __future__ import annotations

import random
import collections
import msgpack
from dataclasses import dataclass
from typing import NamedTuple

import chess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from board_encoder import board_to_tensor, move_to_action, NUM_ACTIONS
from network import ChessNet, AlphaZeroLoss, count_parameters
from mcts import MCTS

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Network
    num_res_blocks: int = 10
    channels: int = 256

    # Training
    buffer_size: int = 100_000
    batch_size: int = 100
    train_steps: int = 100
    lr: float = 1e-3
    num_iterations: int = 4000
    checkpoint_every: int = 200

    # Self-play
    games_per_iter: int = 10
    max_game_moves: int = 200
    num_sims: int = 50 # MCTS
    
    # Dataset
    active_files: int = 4          # how many shard files to mix at once
    samples_per_file: int = 2500   # active_files * samples_per_file ≈ max_samples
    max_samples: int = active_files * samples_per_file
    top_k_moves: int = 3
    path: str = r'C:\Users\login\tree_fish\tree_fish\backend\data'
    
    # Misc
    device: str = "cuda"
    seed: int = 72

cfg = Config()

# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def plot_loss(loss_history: list, policy_history: list, value_history: list):
    plt.figure()
    plt.ion()
    plt.clf()
    plt.plot(loss_history, label="Total")
    plt.plot(policy_history, label="Policy")
    plt.plot(value_history, label="Value")
    plt.legend()
    plt.pause(1)

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class GameSample(NamedTuple):
    board_tensor: torch.Tensor  # (18,8,8)
    mcts_policy: torch.Tensor   # (NUM_ACTIONS,)
    outcome: float              # +1/0/-1


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer: collections.deque[GameSample] = collections.deque(maxlen=max_size)
    
    def add(self, samples: list[GameSample]):
        self.buffer.extend(samples)
    
    def sample(self, n: int) -> list[GameSample]:
        return random.sample(self.buffer, min(n, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class SampleDataset(Dataset):
    def __init__(self, samples: list[GameSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s.board_tensor, s.mcts_policy, torch.tensor([s.outcome], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Self-play trianing utils
# ---------------------------------------------------------------------------

def play_game(net: ChessNet, mcts: MCTS, device: torch.device) -> list[GameSample]:
    """Play one self-play game and return training samples."""
    board = chess.Board()
    history: list[tuple[chess.Board, torch.Tensor]] = []
    
    move_num = 0
    while not board.is_game_over() and move_num < cfg.max_game_moves:
        # Use temperature=1 for first 30 moves, then 0 (deterministic)
        mcts.temperature = 1.0 if move_num < 30 else 0.0
        
        # Run MCTS
        root = mcts.run(board, add_noise=True)
        moves, probs = mcts.get_policy(root)
        
        # Store MCTS policy for this position
        policy_vec = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        for m, p in zip(moves, probs):
            policy_vec[move_to_action(m)] = p
        history.append((board.copy(stack=False), policy_vec))
        
        # Sample move
        if mcts.temperature > 0:
            chosen = np.random.choice(len(moves), p=probs)
        else:
            chosen = int(probs.argmax())
        
        board.push(moves[chosen])
        move_num += 1
    
    # Determine game outcome
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        z_white = 0.0  # draw
    else:
        z_white = 1.0 if outcome.winner == chess.WHITE else -1.0
    
    # Create training samples with outcome labels
    samples = []
    for ply_board, policy in history:
        # Outcome from perspective of player to move
        z = z_white if ply_board.turn == chess.WHITE else -z_white
        tensor = board_to_tensor(ply_board, device=device)
        samples.append(GameSample(tensor.cpu(), policy, z))
    
    result = "draw" if z_white == 0 else ("1-0" if z_white > 0 else "0-1")
    print(f"    Game: {move_num} moves, result={result}")
    
    return samples

# ---------------------------------------------------------------------------
# Dataset training utils
# ---------------------------------------------------------------------------

def chessbench_record_to_sample(
    record: dict,
    records_used: int,
    tau: float = 0.05,
) -> GameSample:
    """
    Convert a ChessBench record into a GameSample.
    
    record format:
    {
        "fen": str,
        "moves": {
            "e2e4": {"win_prob": float},
            ...
        }
    }
    """
    import chess
    
    board = chess.Board(record["fen"])
    
    # --- 1. Encode board ---
    board_tensor = board_to_tensor(board, device=torch.device("cpu"))
    
    # --- 2. Build policy vector ---
    policy_vec = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    
    moves = []
    scores = []
    
    for move_uci, info in record["moves"].items():
        move = chess.Move.from_uci(move_uci)
        
        if move not in board.legal_moves:
            continue  # safety
        
        win_prob = info["win_prob"]
        
        # Convert win_prob → value in [-1, 1]
        value = 2 * win_prob - 1
        
        moves.append(move)
        scores.append(value)
    
    if len(scores) == 0:
        # fallback (shouldn’t happen often)
        return None
    
    scores = torch.tensor(scores, dtype=torch.float32)
    
    if records_used <= 1_000_000:
        # --- 3. Softmax over moves ---
        probs = torch.softmax(scores / tau * 2, dim=0)
        
        for move, p in zip(moves, probs):
            action = move_to_action(move)
            policy_vec[action] = p
    elif records_used <= 5_000_000:
        # --- keep only top-k moves ---
        k = min(10, scores.numel())
        scores, topk_idx = torch.topk(scores, k=k, dim=0)

        # --- softmax over top-k only ---
        probs = torch.softmax(scores / tau, dim=0)

        for idx, p in zip(topk_idx, probs):
            move = moves[idx]
            action = move_to_action(move)
            policy_vec[action] = p
    else:
        # --- keep only top-k moves ---
        k = min(cfg.top_k_moves, scores.numel())
        scores, topk_idx = torch.topk(scores, k=k, dim=0)

        # --- softmax over top-k only ---
        probs = torch.softmax(scores / (tau / 2), dim=0)

        for idx, p in zip(topk_idx, probs):
            move = moves[idx]
            action = move_to_action(move)
            policy_vec[action] = p
    
    # --- 4. Compute value target ---
    # Expected value under the distribution
    outcome = torch.sum(probs * scores).item()
    
    return GameSample(board_tensor, policy_vec, outcome)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    net: ChessNet,
    optimizer: optim.Optimizer,
    criterion: AlphaZeroLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch."""
    net.train()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    n_batches = 0
    
    for boards, policies, values in loader:
        boards = boards.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # Forward pass
        policy_logits, value_pred = net(boards)
        loss, p_loss, v_loss = criterion(policy_logits, value_pred, policies, values)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_policy += p_loss.item()
        total_value += v_loss.item()
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "policy": total_policy / n_batches,
        "value": total_value / n_batches,
    }

# ---------------------------------------------------------------------------
# Main training loops
# ---------------------------------------------------------------------------

def train_self_play():
    # Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build network
    net = ChessNet(num_res_blocks=cfg.num_res_blocks, channels=cfg.channels).to(device)
    print(f"Parameters: {count_parameters(net):,}\n")
    
    # Optimizer and loss
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_iterations)
    criterion = AlphaZeroLoss()
    
    # Replay buffer and MCTS
    buffer = ReplayBuffer(cfg.buffer_size)
    mcts = MCTS(net, device, num_sims=cfg.num_sims)
    
    print(f"Config: {cfg.num_sims} sims/move, {cfg.games_per_iter} games/iter\n")
    
    # Training loop
    for iteration in range(1, cfg.num_iterations + 1):
        print(f"{'='*60}")
        print(f"Iteration {iteration}/{cfg.num_iterations}")
        print(f"{'='*60}")
        
        # Self-play phase
        print(f"Self-play ({cfg.games_per_iter} games):")
        new_samples = []
        for game_num in range(cfg.games_per_iter):
            samples = play_game(net, mcts, device)
            new_samples.extend(samples)
        
        buffer.add(new_samples)
        print(f"  Added {len(new_samples)} positions, buffer size = {len(buffer)}\n")
        
        # Training phase
        if len(buffer) < cfg.batch_size:
            print("Buffer too small, skipping training\n")
            continue
        
        print(f"Training ({cfg.train_steps} steps):")
        batch_samples = buffer.sample(cfg.train_steps * cfg.batch_size)
        dataset = SampleDataset(batch_samples)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        
        metrics = train(net, optimizer, criterion, loader, device)
        scheduler.step()
        
        print(f"  loss={metrics['loss']:.4f}  "
              f"policy={metrics['policy']:.4f}  "
              f"value={metrics['value']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}\n")
        
        # Checkpoint
        if iteration % cfg.checkpoint_every == 0:
            path = f"checkpoint_iter{iteration:04d}.pt"
            torch.save({
                "iteration": iteration,
                "num_res_blocks": cfg.num_res_blocks,
                "channels": cfg.channels,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, path)
            print(f"  Checkpoint saved: {path}\n")
    
    print("Training complete!")

def train_on_dataset_from_loaded_checkpoint(path: str):
    import glob
    from pathlib import Path
    import time
    
    loss_history = []
    policy_history = []
    value_history = []

    # Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print(f"Loading checkpoint: {path}")
    
    # Try new format first (weights_only safe mode)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        raise Exception("Path points to an invalid/nonexistent checkpoint")
    
    net = ChessNet(num_res_blocks=cfg.num_res_blocks, channels=cfg.channels)
    net.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    print(f"Parameters: {count_parameters(net):,}\n")
    
    # Optimizer and loss
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_iterations)
    criterion = AlphaZeroLoss()
    
    # Replay buffer
    buffer = ReplayBuffer(cfg.buffer_size)
    
    # Find all msgpack files in the directory
    data_dir = Path(cfg.path)
    msgpack_files = sorted(glob.glob(str(data_dir / "*.msgpack")))
    print(f"Found {len(msgpack_files)} msgpack files in {data_dir}")
    print(f"Config: {int(cfg.max_samples)} samples per iteration\n")
    
       # Shuffle file order once at the start
    random.shuffle(msgpack_files)

    # State for each active file stream
    stream_states = []
    next_file_idx = 0

    def open_new_stream():
        nonlocal next_file_idx

        # wrap around and reshuffle after a full pass
        if next_file_idx >= len(msgpack_files):
            next_file_idx = 0
            random.shuffle(msgpack_files)

        filepath = msgpack_files[next_file_idx]
        next_file_idx += 1

        fh = open(filepath, "rb")
        unpacker = msgpack.Unpacker(fh, raw=False)

        return {
            "filepath": filepath,
            "handle": fh,
            "unpacker": unpacker,
        }

    # initialize active streams
    num_streams = min(cfg.active_files, len(msgpack_files))
    for _ in range(num_streams):
        stream_states.append(open_new_stream())

    def refill_stream(i: int):
        old = stream_states[i]
        old["handle"].close()
        stream_states[i] = open_new_stream()

    def get_mixed_samples(total_n: int) -> list[GameSample]:
        samples = []

        if len(stream_states) == 0:
            return samples

        per_stream = max(1, total_n // len(stream_states))

        # read roughly evenly from each active stream
        for i in range(len(stream_states)):
            taken = 0

            while taken < per_stream and len(samples) < total_n:
                stream = stream_states[i]

                try:
                    record = next(stream["unpacker"])
                except StopIteration:
                    refill_stream(i)
                    continue

                sample = chessbench_record_to_sample(record)
                if sample is None:
                    continue

                samples.append(sample)
                taken += 1

        # if integer division left us short, top up from random active streams
        while len(samples) < total_n and len(stream_states) > 0:
            i = random.randrange(len(stream_states))
            stream = stream_states[i]

            try:
                record = next(stream["unpacker"])
            except StopIteration:
                refill_stream(i)
                continue

            sample = chessbench_record_to_sample(record)
            if sample is None:
                continue

            samples.append(sample)

        random.shuffle(samples)
        return samples
    
    # Training loop
    for iteration in range(1, cfg.num_iterations + 1):
        print(f"{'='*60}")
        print(f"Iteration {iteration}/{cfg.num_iterations}")
        print(f"{'='*60}")
        
        # Load next chunk of samples
        load_start = time.time()
        new_samples = get_mixed_samples(int(cfg.max_samples))
        load_time = time.time() - load_start
        
        if len(new_samples) == 0:
            print("No more samples available in dataset!")
            break
        
        buffer.add(new_samples)
        print(f"  Loaded {len(new_samples)} positions in {load_time:.2f}s, buffer size = {len(buffer)}\n")
        
        # Training phase
        if len(buffer) < cfg.batch_size:
            print("Buffer too small, skipping training\n")
            continue
        
        print(f"Training ({cfg.train_steps} steps):")
        batch_samples = buffer.sample(cfg.train_steps * cfg.batch_size)
        dataset = SampleDataset(batch_samples)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        
        # Train with timing
        train_start = time.time()
        metrics = train(net, optimizer, criterion, loader, device)
        train_time = time.time() - train_start
        scheduler.step()

        loss_history.append(metrics["loss"])
        policy_history.append(metrics["policy"])
        value_history.append(metrics["value"])
        
        # Calculate batch timing
        n_batches = len(batch_samples) // cfg.batch_size
        ms_per_batch = (train_time / n_batches) * 1000 if n_batches > 0 else 0
        
        print(f"  loss={metrics['loss']:.4f}  "
              f"policy={metrics['policy']:.4f}  "
              f"value={metrics['value']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"  Train time: {train_time:.2f}s ({n_batches} batches, {ms_per_batch:.1f}ms/batch)\n")
        
        # Checkpoint
        if iteration % cfg.checkpoint_every == 0:
            path = f"checkpoint_iter{iteration:04d}.pt"
            torch.save({
                "iteration": iteration,
                "num_res_blocks": cfg.num_res_blocks,
                "channels": cfg.channels,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, path)
            print(f"  Checkpoint saved: {path}\n")
    
    # Cleanup
    for stream in stream_states:
        stream["handle"].close()
    
    plot_loss(loss_history, policy_history, value_history)

    print("Training complete!")

def train_on_dataset():
    import glob
    from pathlib import Path
    import time
    
    loss_history = []
    policy_history = []
    value_history = []
    records_used = 0

    # Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build network
    net = ChessNet(num_res_blocks=cfg.num_res_blocks, channels=cfg.channels).to(device)
    print(f"Parameters: {count_parameters(net):,}\n")
    
    # Optimizer and loss
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_iterations)
    criterion = AlphaZeroLoss()
    
    # Replay buffer
    buffer = ReplayBuffer(cfg.buffer_size)
    
    # Find all msgpack files in the directory
    data_dir = Path(cfg.path)
    msgpack_files = sorted(glob.glob(str(data_dir / "*.msgpack")))
    print(f"Found {len(msgpack_files)} msgpack files in {data_dir}")
    print(f"Config: {int(cfg.max_samples)} samples per iteration\n")
    
       # Shuffle file order once at the start
    random.shuffle(msgpack_files)

    # State for each active file stream
    stream_states = []
    next_file_idx = 0
    records_used = 0

    def open_new_stream():
        nonlocal next_file_idx

        # wrap around and reshuffle after a full pass
        if next_file_idx >= len(msgpack_files):
            next_file_idx = 0
            random.shuffle(msgpack_files)

        filepath = msgpack_files[next_file_idx]
        next_file_idx += 1

        fh = open(filepath, "rb")
        unpacker = msgpack.Unpacker(fh, raw=False)

        return {
            "filepath": filepath,
            "handle": fh,
            "unpacker": unpacker,
        }

    # initialize active streams
    num_streams = min(cfg.active_files, len(msgpack_files))
    for _ in range(num_streams):
        stream_states.append(open_new_stream())

    def refill_stream(i: int):
        old = stream_states[i]
        old["handle"].close()
        stream_states[i] = open_new_stream()

    def get_mixed_samples(total_n: int) -> list[GameSample]:
        nonlocal records_used
        samples = []

        if len(stream_states) == 0:
            return samples

        per_stream = max(1, total_n // len(stream_states))

        # read roughly evenly from each active stream
        for i in range(len(stream_states)):
            taken = 0

            while taken < per_stream and len(samples) < total_n:
                stream = stream_states[i]

                try:
                    record = next(stream["unpacker"])
                except StopIteration:
                    refill_stream(i)
                    continue

                sample = chessbench_record_to_sample(record, records_used)
                records_used += 1
                if sample is None:
                    continue

                samples.append(sample)
                taken += 1

        # if integer division left us short, top up from random active streams
        while len(samples) < total_n and len(stream_states) > 0:
            i = random.randrange(len(stream_states))
            stream = stream_states[i]

            try:
                record = next(stream["unpacker"])
            except StopIteration:
                refill_stream(i)
                continue

            sample = chessbench_record_to_sample(record)
            if sample is None:
                continue

            samples.append(sample)

        random.shuffle(samples)
        return samples
    
    # Training loop
    for iteration in range(1, cfg.num_iterations + 1):
        print(f"{'='*60}")
        print(f"Iteration {iteration}/{cfg.num_iterations}")
        print(f"{'='*60}")
        
        # Load next chunk of samples
        load_start = time.time()
        new_samples = get_mixed_samples(int(cfg.max_samples))
        load_time = time.time() - load_start
        
        if len(new_samples) == 0:
            print("  No more samples available in dataset!")
            break
        
        buffer.add(new_samples)
        print(f"  Loaded {len(new_samples)} positions in {load_time:.2f}s, buffer size = {len(buffer)}\n")
        
        # Training phase
        if len(buffer) < cfg.batch_size:
            print("Buffer too small, skipping training\n")
            continue
        
        print(f"Training ({cfg.train_steps} steps):")
        batch_samples = buffer.sample(cfg.train_steps * cfg.batch_size)
        dataset = SampleDataset(batch_samples)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        
        # Train with timing
        train_start = time.time()
        metrics = train(net, optimizer, criterion, loader, device)
        train_time = time.time() - train_start
        scheduler.step()

        loss_history.append(metrics["loss"])
        policy_history.append(metrics["policy"])
        value_history.append(metrics["value"])
        
        # Calculate batch timing
        n_batches = len(batch_samples) // cfg.batch_size
        ms_per_batch = (train_time / n_batches) * 1000 if n_batches > 0 else 0
        
        print(f"  loss={metrics['loss']:.4f}  "
              f"policy={metrics['policy']:.4f}  "
              f"value={metrics['value']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"  Train time: {train_time:.2f}s ({n_batches} batches, {ms_per_batch:.1f}ms/batch)\n")
        
        # Checkpoint
        if iteration % cfg.checkpoint_every == 0:
            path = f"checkpoint_iter{iteration:04d}.pt"
            torch.save({
                "iteration": iteration,
                "num_res_blocks": cfg.num_res_blocks,
                "channels": cfg.channels,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, path)
            print(f"  Checkpoint saved: {path}\n")
    
    # Cleanup
    for stream in stream_states:
        stream["handle"].close()
    
    plot_loss(loss_history, policy_history, value_history)

    print("Training complete!")

def main():
    train_on_dataset()
    # train_on_dataset_from_loaded_checkpoint(r"C:\Users\login\tree_fish\tree_fish\checkpoint_iter0500.pt")

if __name__ == "__main__":
    main()