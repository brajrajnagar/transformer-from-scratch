"""
Transformer from Scratch - Lesson 4: Encoder Layer
===================================================

Now we combine all the pieces we've learned into the ENCODER layer.

WHAT WE'VE BUILT SO FAR:
  1. Token + Position Embeddings
  2. Multi-Head Self-Attention

TODAY'S PLACE IN PIPELINE:
  Input embeddings → [Encoder Layer × N] → Encoder Output

WHAT WE'LL BUILD:
  1. Multi-Head Self-Attention block
  2. Feed-Forward Network (FFN)
  3. Residual Connection + Layer Normalization
  4. Complete Encoder Layer
  5. Stack of Encoder Layers
"""

import math
import torch
import torch.nn as nn


# ============================================================================
# COMPONENT 1: Feed-Forward Network (FFN)
# ============================================================================

class FeedForwardNetwork(nn.Module):
    """
    The position-wise Feed-Forward Network in each encoder layer.

    The FFN is applied to EACH position independently and identically.

    ARCHITECTURE:
    ─────────────
    Input:  (batch, seq_len, d_model)
      ↓
    Linear 1: d_model → d_ff       (expand)
      ↓
    ReLU activation
      ↓
    Linear 2: d_ff → d_model       (project back)
      ↓
    Output: (batch, seq_len, d_model)

    DIMENSIONS (original paper):
      d_model = 512, d_ff = 2048

    WHY d_ff > d_model?
    ───────────────────
    Expanding to a higher dimension gives the network more capacity
    to learn complex transformations, then projects back down.
    It's like a bottleneck layer.

    NOTE: This is "position-wise" because the SAME FFN is applied to
    every position independently. Position 0 and Position 1 both use
    the exact same weights.
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # Expand
            nn.ReLU(),                   # Activation
            nn.Dropout(dropout),         # Regularization
            nn.Linear(d_ff, d_model),   # Project back
            nn.Dropout(dropout),         # Regularization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
        Returns:
            Output tensor, shape (batch, seq_len, d_model)
        """
        return self.net(x)


# ============================================================================
# COMPONENT 2: Layer Normalization
# ============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization: Normalize across the feature dimension.

    WHY LAYER NORM INSTEAD OF BATCH NORM?
    ──────────────────────────────────────
    1. Works well with small batch sizes (even batch_size=1)
    2. Normalizes per-sample, not across batch
    3. More stable for sequence models where seq_len varies

    THE FORMULA:
    ────────────
    For each sample independently:
      μ = mean(features)
      σ² = variance(features)
      normed = (features - μ) / sqrt(σ² + ε)
      output = γ * normed + β

    Where γ (gamma) and β (beta) are learnable parameters.

    INPUTS argument in __init__:
    ───────────────────────────
    INPUTS is the dimension to normalize OVER.
    In the Transformer, we normalize over the d_model (feature) dimension.

    For the Encoder: normalize over d_model for each position
    For the Decoder: same, normalize over d_model for each position

    Typical values:
      - Encoder: normalize over d_model at each position
      - Decoder: same
    """

    def __init__(self, sizes: tuple[int, ...], eps: float = 1e-5):
        super().__init__()

        # Create learnable parameters: gamma (scale) and beta (shift)
        # These let the network UN-normalize if needed
        self.gamma = nn.Parameter(torch.ones(sizes))  # Scale
        self.beta = nn.Parameter(torch.zeros(sizes))     # Shift
        self.eps = eps  # Small constant for numerical stability

        self.sizes = sizes  # Store for forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (*, *self.sizes)
               where * means any number of batch dimensions
        Returns:
            Normalized tensor, same shape as input
        """
        # Compute mean and variance over the LAST dimensions (self.sizes)
        # keepdim=True keeps the dimension so broadcasting works
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / (std + self.eps)

        # Scale and shift
        return x_norm * self.gamma + self.beta


# ============================================================================
# COMPONENT 3: Residual Connection
# ============================================================================

# In the Transformer, residual connections are simple:
#   output = input + sublayer(input)
# But they're ALWAYS followed by LayerNorm:
#   output = LayerNorm(input + sublayer(input))
# This is called "Post-LayerNorm" (used in the original paper).
# Some later work uses "Pre-LayerNorm" (LayerNorm before sublayer).


# ============================================================================
# MAIN: Encoder Layer
# ============================================================================

class EncoderLayer(nn.Module):
    """
    A single Transformer Encoder Layer.

    ARCHITECTURE:
    ─────────────
    Input: (batch, seq_len, d_model)
      ↓
    ┌─────────────────────────────────────────┐
    │ Multi-Head Self-Attention               │
    │   - Each token attends to ALL tokens    │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Residual Connection: x + attention    │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ LayerNorm                               │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Feed-Forward Network                    │
    │   - Applied position-wise               │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Residual Connection: x + FFN          │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ LayerNorm                               │
    └─────────────────────────────────────────┘
          ↓
    Output: (batch, seq_len, d_model)

    THE ORDER (Post-LayerNorm, original paper):
      x → Sublayer → +x → LayerNorm

    SOME IMPLEMENTATIONS use Pre-LayerNorm:
      x → LayerNorm → Sublayer → +x

    We follow the ORIGINAL paper (Post-LayerNorm).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed-Forward Network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

        # Layer Normalization (after each sublayer)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
        Returns:
            Output tensor, shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # ---- Sublayer 1: Multi-Head Self-Attention ----
        # Self-attention: Q = K = V = x
        attn_output, attn_weights = self.self_attention(x, x, x)

        # Residual connection + LayerNorm
        # x + attn_output → normalize
        x = self.norm1(x + self.dropout(attn_output))

        # ---- Sublayer 2: Feed-Forward Network ----
        # FFN is applied position-wise (same weights for each position)
        ffn_output = self.feed_forward(x)

        # Residual connection + LayerNorm
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# We need to import MultiHeadAttention from lesson 3.
# Let's define a minimal version here for standalone execution:
class MultiHeadAttention(nn.Module):
    """Minimal Multi-Head Attention (from Lesson 3) for standalone use."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        b, s, _ = x.shape
        x = x.view(b, s, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        b, h, s, d = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(b, s, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self._split_heads(self.W_Q(Q))
        K = self._split_heads(self.W_K(K))
        V = self._split_heads(self.W_V(V))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attention_dropout(attn)
        output = self._merge_heads(torch.matmul(attn, V))
        output = self.W_O(output)
        output = self.output_dropout(output)
        return output, attn


def demonstrate_feed_forward_network():
    """Show how the Feed-Forward Network works."""
    print("=" * 70)
    print("FEED-FORWARD NETWORK DEMO")
    print("=" * 70)
    print()

    d_model = 16
    d_ff = 64  # Expanded dimension
    batch_size = 2
    seq_len = 3

    ffn = FeedForwardNetwork(d_model, d_ff, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    output = ffn(x)
    print(f"Output shape: {output.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    print(f"Architecture:")
    print(f"  Linear:  {d_model} → {d_ff}")
    print(f"  ReLU")
    print(f"  Dropout ({0.0})")
    print(f"  Linear:  {d_ff} → {d_model}")
    print()

    print(f"Key property: Position-wise!")
    print(f"  Position 0 and Position 1 use the EXACT SAME weights.")
    print(f"  The FFN processes each position independently.")
    print()


def demonstrate_layer_norm():
    """Show how Layer Normalization works."""
    print("=" * 70)
    print("LAYER NORMALIZATION DEMO")
    print("=" * 70)
    print()

    batch_size = 2
    seq_len = 3
    d_model = 4

    layer_norm = LayerNorm((d_model,))

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    # Show what gets normalized
    print("LayerNorm normalizes ACROSS the feature dimension:")
    for b in range(batch_size):
        for s in range(seq_len):
            vals = x[b, s].tolist()
            mean = x[b, s].mean().item()
            std = x[b, s].std().item()
            print(f"  Position [{b},{s}]: {vals}")
            print(f"    mean={mean:.4f}, std={std:.4f}")
    print()

    output = layer_norm(x)
    print("After LayerNorm:")
    for b in range(batch_size):
        for s in range(seq_len):
            vals = output[b, s].tolist()
            mean = output[b, s].mean().item()
            std = output[b, s].std().item()
            print(f"  Position [{b},{s}]: {[f'{v:.4f}' for v in vals]}")
            print(f"    mean≈{mean:.6f}, std≈{std:.4f}")
    print()
    print("✓ All positions now have mean≈0 and std≈1")


def demonstrate_encoder_layer():
    """Show a complete encoder layer in action."""
    print("=" * 70)
    print("ENCODER LAYER DEMO")
    print("=" * 70)
    print()

    d_model = 64
    n_heads = 4
    d_ff = 128
    batch_size = 2
    seq_len = 5

    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    print("Processing through encoder layer:")
    print(f"  1. Self-Attention: each token attends to all tokens")
    print(f"  2. Residual + LayerNorm")
    print(f"  3. Feed-Forward: position-wise processing")
    print(f"  4. Residual + LayerNorm")
    print()

    output = encoder_layer(x)
    print(f"Output shape: {output.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    print("Key observations:")
    print("  ✓ Input and output have the SAME shape")
    print("  ✓ Each position now has CONTEXT from all other positions")
    print("  ✓ The model has learned to combine information across the sequence")
    print()


def demonstrate_full_encoder():
    """Show a stack of encoder layers."""
    print("=" * 70)
    print("FULL ENCODER (Stack of N Layers)")
    print("=" * 70)
    print()

    d_model = 64
    n_heads = 4
    d_ff = 128
    n_layers = 3  # Using 3 for demo (original paper uses 6)
    batch_size = 1
    seq_len = 4

    encoder = nn.ModuleList([
        EncoderLayer(d_model, n_heads, d_ff, dropout=0.0)
        for _ in range(n_layers)
    ])

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Encoder: {n_layers} stacked layers")
    print(f"Input shape: {x.shape}")
    print()

    for i, layer in enumerate(encoder):
        x = layer(x)
        print(f"After encoder layer {i+1}: {x.shape}")

    print()
    print(f"Final encoder output: {x.shape}")
    print()
    print("Each layer builds on top of the previous one:")
    print("  Layer 1: Direct context from input tokens")
    print("  Layer 2: Context from context (2-hop relationships)")
    print("  Layer 3: Context from context from context (3-hop)")
    print("  ...and so on → deeper layers capture more abstract relationships")
    print()


def show_encoder_parameters():
    """Show parameter counts for different configurations."""
    print("=" * 70)
    print("ENCODER PARAMETER COUNTS")
    print("=" * 70)
    print()

    configs = [
        ("Tiny", 64, 4, 128, 2),
        ("Small", 256, 8, 512, 4),
        ("Base", 512, 8, 2048, 6),
        ("Big", 1024, 16, 4096, 6),
    ]

    print(f"{'Config':<10} {'d_model':<10} {'heads':<8} {'d_ff':<10} {'layers':<8} {'Params (M)':<12}")
    print("-" * 70)

    for name, d_model, n_heads, d_ff, n_layers in configs:
        layer = EncoderLayer(d_model, n_heads, d_ff, dropout=0.1)
        n_params = sum(p.numel() for p in layer.parameters())
        print(f"{name:<10} {d_model:<10} {n_heads:<8} {d_ff:<10} {n_layers:<8} {n_params/1e6:<12.2f}")

    print()
    print("Original Transformer (Base): 6 layers, d_model=512 → ~10M params per encoder layer")
    print("Original Transformer (Big):  6 layers, d_model=1024 → ~45M params per encoder layer")
    print()


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "TRANSFORMER FROM SCRATCH - LESSON 4" + " " * 20 + "║")
    print("║" + " " * 20 + "Encoder Layer" + " " * 36 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Demo 1: Feed-Forward Network
    demonstrate_feed_forward_network()
    print()

    # Demo 2: Layer Normalization
    demonstrate_layer_norm()
    print()

    # Demo 3: Encoder Layer
    demonstrate_encoder_layer()
    print()

    # Demo 4: Full Encoder
    demonstrate_full_encoder()
    print()

    # Parameter counts
    show_encoder_parameters()
    print()

    print("=" * 70)
    print("SUMMARY OF LESSON 4:")
    print("-" * 70)
    print("✓ Feed-Forward Network: d_model → d_ff → d_model (position-wise)")
    print("✓ Layer Normalization: normalize per-sample across features")
    print("✓ Residual Connection: x + sublayer(x)")
    print("✓ Encoder Layer: Self-Attention → Residual+LN → FFN → Residual+LN")
    print("✓ Stack N encoder layers for deeper processing")
    print()
    print("ENCODER LAYER FORMULA:")
    print("  x → LayerNorm(x + MultiHeadSelfAttention(x))")
    print("  → LayerNorm(x + FFN(x))")
    print()
    print("NEXT: Decoder Layer (masked attention + cross-attention)")
    print("Run: python 05_decoder_layer.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()