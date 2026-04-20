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
from dataclasses import dataclass
from typing import NamedTuple

import chess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from board_encoder import board_to_tensor, move_to_action, NUM_ACTIONS
from network import ChessNet, AlphaZeroLoss, count_parameters
from mcts_simple import SimpleMCTS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Network
    num_res_blocks: int = 4
    channels: int = 128
    
    # MCTS
    num_sims: int = 50
    
    # Training
    games_per_iter: int = 10
    max_game_moves: int = 200
    buffer_size: int = 10000
    batch_size: int = 128
    train_steps: int = 100
    lr: float = 1e-3
    num_iterations: int = 50
    checkpoint_every: int = 5
    
    # Misc
    device: str = "cuda"
    seed: int = 42


cfg = Config()


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
# Self-play
# ---------------------------------------------------------------------------

def play_game(net: ChessNet, mcts: SimpleMCTS, device: torch.device) -> list[GameSample]:
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
# Main training loop
# ---------------------------------------------------------------------------

def main():
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
    mcts = SimpleMCTS(net, device, num_sims=cfg.num_sims)
    
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


if __name__ == "__main__":
    main()