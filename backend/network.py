"""
network.py
----------
AlphaZero-style dual-headed residual network for chess.

Architecture
------------
Input  : (B, 18, 8, 8)  — 18-plane board encoding
         ↓
Stem   : Conv(18→256, 3×3) + BN + ReLU
         ↓
Tower  : N × ResidualBlock(256 channels)
         ↓
       ┌─────────────────────┬─────────────────────┐
Policy head                        Value head
Conv(256→2, 1×1)+BN+ReLU     Conv(256→1, 1×1)+BN+ReLU
Flatten → Linear(128→4672)   Flatten → Linear(64) → ReLU
Softmax (or raw logits)       → Linear(1) → Tanh  →  v ∈ [-1,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from board_encoder import NUM_PLANES, NUM_ACTIONS, BOARD_SIZE


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Standard pre-activation residual block used in AlphaZero:
        Conv(3×3) → BN → ReLU → Conv(3×3) → BN
        + skip connection
        → ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual, inplace=True)
        return out


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class ChessNet(nn.Module):
    """
    Policy-value network for chess MCTS.

    Args:
        num_res_blocks : depth of the residual tower (AlphaZero uses 19 or 39)
        channels       : width of every residual block
        policy_channels: channels in the policy head conv (AlphaZero uses 2)
        value_channels : channels in the value  head conv (AlphaZero uses 1)
        value_hidden   : hidden size of the value MLP
    """

    def __init__(
        self,
        num_res_blocks: int = 10,
        channels: int = 256,
        policy_channels: int = 2,
        value_channels: int = 1,
        value_hidden: int = 256,
    ):
        super().__init__()

        # ── Stem ────────────────────────────────────────────────────────────
        self.stem = ConvBnRelu(NUM_PLANES, channels, kernel=3, padding=1)

        # ── Residual tower ──────────────────────────────────────────────────
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        board_squares = BOARD_SIZE * BOARD_SIZE  # 64

        # ── Policy head ─────────────────────────────────────────────────────
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, policy_channels, 1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(policy_channels * board_squares, NUM_ACTIONS)

        # ── Value head ──────────────────────────────────────────────────────
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_channels, 1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(inplace=True),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(value_channels * board_squares, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, 18, 8, 8) float tensor

        Returns:
            policy_logits : (B, NUM_ACTIONS=4672) — raw logits (use F.softmax for probs)
            value         : (B, 1)                — position eval in [-1, 1]
        """
        # Shared trunk
        h = self.stem(x)
        h = self.tower(h)

        # Policy head
        p = self.policy_conv(h)
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_conv(h)
        v = v.flatten(start_dim=1)
        value = self.value_mlp(v)

        return policy_logits, value

    # ------------------------------------------------------------------
    # Convenience: single-board inference (no batch dim)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        board_tensor: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Run a single (18, 8, 8) board tensor through the network.

        Args:
            board_tensor : (18, 8, 8) float tensor
            legal_mask   : (NUM_ACTIONS,) bool tensor — if provided, illegal
                           moves are masked to -∞ before softmax

        Returns:
            policy_probs : (NUM_ACTIONS,) float tensor summing to 1
            value        : Python float in [-1, 1]
        """
        self.eval()
        x = board_tensor.unsqueeze(0)  # → (1, 18, 8, 8)
        logits, val = self(x)
        logits = logits.squeeze(0)    # → (4672,)

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))

        policy_probs = F.softmax(logits, dim=0)
        return policy_probs, val.item()


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class AlphaZeroLoss(nn.Module):
    """
    Combined policy + value loss as used in AlphaZero:

        L = MSE(v, z) - π^T · log(p) + λ · ||θ||²

    where z ∈ {-1, 0, 1} is the game outcome (from the current player's POV),
    π is the MCTS visit-count distribution, and p is the network policy.
    L2 regularisation is handled by the optimiser (weight_decay), so we omit it here.

    Args:
        value_weight  : relative weight of the value loss (default 1.0)
        policy_weight : relative weight of the policy loss (default 1.0)
    """

    def __init__(self, value_weight: float = 1.0, policy_weight: float = 1.0):
        super().__init__()
        self.value_weight  = value_weight
        self.policy_weight = policy_weight

    def forward(
        self,
        policy_logits: torch.Tensor,  # (B, NUM_ACTIONS)
        value_pred:    torch.Tensor,  # (B, 1)
        target_policy: torch.Tensor,  # (B, NUM_ACTIONS) — MCTS visit distribution
        target_value:  torch.Tensor,  # (B, 1)           — game outcome z ∈ [-1,1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, policy_loss, value_loss
        """
        value_loss  = F.mse_loss(value_pred, target_value)
        # Cross-entropy with soft MCTS target: -sum(π · log(p))
        log_probs   = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
        total_loss  = self.value_weight * value_loss + self.policy_weight * policy_loss
        return total_loss, policy_loss, value_loss


# ---------------------------------------------------------------------------
# Tiny convenience: model summary
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import chess
    from board_encoder import board_to_tensor, legal_moves_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    net = ChessNet(num_res_blocks=10, channels=256).to(device)
    print(f"Parameters: {count_parameters(net):,}")

    # Quick forward pass on the starting position
    board = chess.Board()
    tensor = board_to_tensor(board, device=device)
    mask   = legal_moves_mask(board, device=device)
    probs, val = net.predict(tensor, mask)

    print(f"Policy distribution over {mask.sum().item()} legal moves")
    print(f"Position value (White POV): {val:.4f}")
    print(f"Top-5 action indices: {probs.topk(5).indices.tolist()}")
