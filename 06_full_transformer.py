"""
Transformer from Scratch - Lesson 6: Full Transformer Model
============================================================

All components are now combined into the complete Transformer model!

WHAT WE'VE BUILT SO FAR:
  1. Token + Position Embeddings
  2. Multi-Head Self-Attention
  3. Encoder Layer (attention + FFN + residual + layernorm)
  4. Decoder Layer (masked attention + cross-attention + FFN + residual + layernorm)

TODAY: Put it ALL together into a complete encoder-decoder Transformer.

PIPELINE:
  Input: English text "I am a cat"
  Target: French text "<sos> Je suis un chat <eos>"

  ┌──────────────────────────────────────────────────────────────────┐
  │                    TRANSFORMER MODEL                             │
  │                                                                  │
  │  ┌──────────────┐    ┌────────────────────┐    ┌──────────────┐ │
  │  │   EMBEDDING  │───►│   ENCODER          │    │   EMBEDDING  │ │
  │  │ + POSITION   │    │   (N layers)       │    │ + POSITION   │ │
  │  └──────────────┘    │                    │    └──────────────┘ │
  │                      │                    │                     │
  │                      │                    │                     │
  │                      │                    │                     │
  │                      └────────────────────┘────────────────────┘
  │                                    │
  │                                    ▼
  │                      ┌────────────────────┐
  │                      │   DECODER          │
  │                      │   (N layers)       │
  │                      │   (with cross-attn)│
  │                      └────────────────────┘
  │                                    │
  │                                    ▼
  │                      ┌────────────────────┐
  │                      │  Linear + Softmax  │
  │                      └────────────────────┘
  │                                    │
  │                                    ▼
  │                        French probabilities over vocab
  └──────────────────────────────────────────────────────────────────┘
"""

import math
import torch
import torch.nn as nn


# ============================================================================
# COMPONENT 1: Token Embedding
# ============================================================================

class TokenEmbedding(nn.Module):
    """Converts token IDs to dense vectors."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# ============================================================================
# COMPONENT 2: Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encodings to token embeddings."""

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ============================================================================
# COMPONENT 3: Multi-Head Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention (for encoder and masked decoder attention)."""

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


# ============================================================================
# COMPONENT 4: Cross Attention
# ============================================================================

class CrossAttention(nn.Module):
    """Multi-Head Cross-Attention: decoder attends to encoder output."""

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

    def forward(self, decoder_input, encoder_output, mask=None):
        Q = self._split_heads(self.W_Q(decoder_input))
        K = self._split_heads(self.W_K(encoder_output))
        V = self._split_heads(self.W_V(encoder_output))

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


# ============================================================================
# COMPONENT 5: Feed-Forward Network
# ============================================================================

class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# COMPONENT 6: Layer Normalization
# ============================================================================

class LayerNorm(nn.Module):
    """Layer Normalization: normalize across the feature dimension."""

    def __init__(self, sizes: tuple[int, ...], eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(sizes))
        self.beta = nn.Parameter(torch.zeros(sizes))
        self.eps = eps
        self.sizes = sizes

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (std + self.eps)
        return x_norm * self.gamma + self.beta


# ============================================================================
# COMPONENT 7: Encoder Layer
# ============================================================================

class EncoderLayer(nn.Module):
    """A single Transformer Encoder Layer."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


# ============================================================================
# COMPONENT 8: Decoder Layer
# ============================================================================

class DecoderLayer(nn.Module):
    """A single Transformer Decoder Layer."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))
        self.norm3 = LayerNorm((d_model,))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, causal_mask, padding_mask=None):
        attn_output, _ = self.masked_attention(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout(attn_output))

        cross_attn_output, _ = self.cross_attention(x, encoder_output, padding_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ============================================================================
# MAIN: Complete Transformer Model
# ============================================================================

class Transformer(nn.Module):
    """
    The complete Transformer encoder-decoder model.

    ARCHITECTURE:
    ─────────────
    1. ENCODER SIDE:
       Input tokens → Token Embedding → Positional Encoding → [Encoder Layer × N]

    2. DECODER SIDE:
       Target tokens → Token Embedding → Positional Encoding →
       [Decoder Layer × N] → Linear → Softmax

    3. CONNECTION:
       Decoder attends to Encoder output via Cross-Attention

    SHAPES THROUGH THE MODEL:
    ─────────────────────────
    Input tokens:          (batch, src_seq_len)
    Encoder output:        (batch, src_seq_len, d_model)
    Target tokens:         (batch, tgt_seq_len)
    Decoder output:        (batch, tgt_seq_len, d_model)
    Logits:                (batch, tgt_seq_len, vocab_size)
    Probabilities:         (batch, tgt_seq_len, vocab_size)
    """

    def __init__(
        self,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # Embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.src_pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Encoder
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        # Output projection
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create all required masks for the Transformer.

        Args:
            src: Source tokens, shape (batch, src_seq_len)
            tgt: Target tokens, shape (batch, tgt_seq_len)

        Returns:
            src_padding_mask: (batch, 1, 1, src_seq_len)
            tgt_causal_mask: (tgt_seq_len, tgt_seq_len)
            tgt_padding_mask: (batch, 1, 1, tgt_seq_len)
        """
        batch_size = src.shape[0]
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # 1. Source padding mask (mask padding tokens in source)
        # src_pad_idx marks padding tokens (usually 0)
        src_padding_mask = (src == self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_seq_len)

        # 2. Target causal mask (lower triangular - no looking ahead)
        tgt_causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).to(src.device)  # (tgt_seq_len, tgt_seq_len)

        # 3. Target padding mask (mask padding tokens in target)
        tgt_padding_mask = (tgt == self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_seq_len)

        return src_padding_mask, tgt_causal_mask, tgt_padding_mask

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode the source input.

        Args:
            src: Source tokens, shape (batch, src_seq_len)

        Returns:
            Encoder output, shape (batch, src_seq_len, d_model)
        """
        # Embed + positional encoding
        src_embedded = self.src_embedding(src)
        src_embedded = self.src_pos_encoding(src_embedded)

        # Pass through encoder layers
        output = src_embedded
        for layer in self.encoder:
            output = layer(output)

        return output

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_causal_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode the target with encoder output.

        Args:
            tgt: Target tokens, shape (batch, tgt_seq_len)
            encoder_output: From encoder, shape (batch, src_seq_len, d_model)
            src_padding_mask: Mask for source padding (used in cross-attn)
            tgt_causal_mask: Causal mask for target
            tgt_padding_mask: Mask for target padding (used in self-attn)

        Returns:
            Decoder output, shape (batch, tgt_seq_len, d_model)
        """
        # Embed + positional encoding
        tgt_embedded = self.tgt_embedding(tgt)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)

        # Pass through decoder layers
        output = tgt_embedded
        for layer in self.decoder:
            output = layer(output, encoder_output, tgt_causal_mask, src_padding_mask)

        return output

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the complete Transformer.

        Args:
            src: Source tokens, shape (batch, src_seq_len)
            tgt: Target tokens, shape (batch, tgt_seq_len)

        Returns:
            Output logits, shape (batch, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks
        src_padding_mask, tgt_causal_mask, tgt_padding_mask = self.create_masks(src, tgt)

        # Encode
        encoder_output = self.encode(src)

        # Decode
        decoder_output = self.decode(
            tgt, encoder_output, src_padding_mask, tgt_causal_mask, tgt_padding_mask
        )

        # Project to vocabulary
        logits = self.output_linear(decoder_output)

        return logits


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demonstrate_full_transformer():
    """Run the full Transformer on a small example."""
    print("=" * 70)
    print("FULL TRANSFORMER DEMO")
    print("=" * 70)
    print()

    # Small config for demo
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 64
    n_heads = 4
    n_encoder_layers = 2
    n_decoder_layers = 2
    d_ff = 128
    dropout = 0.0

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
    )

    print(f"Model Configuration:")
    print(f"  Source vocab size:    {src_vocab_size}")
    print(f"  Target vocab size:    {tgt_vocab_size}")
    print(f"  d_model:              {d_model}")
    print(f"  n_heads:              {n_heads}")
    print(f"  Encoder layers:       {n_encoder_layers}")
    print(f"  Decoder layers:       {n_decoder_layers}")
    print(f"  d_ff:                 {d_ff}")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:       {total_params:,}")
    print(f"Trainable parameters:   {trainable_params:,}")
    print()

    # Create dummy batch
    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 4

    # Source: English-like tokens (no padding)
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))

    # Target: French-like tokens (with padding at the end)
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

    print(f"Input shapes:")
    print(f"  Source: {src.shape}  (batch={batch_size}, seq_len={src_seq_len})")
    print(f"  Target: {tgt.shape}  (batch={batch_size}, seq_len={tgt_seq_len})")
    print()

    print("Forward pass:")
    print(f"  1. Encode source input → encoder_output")
    print(f"  2. Decode target with encoder output → decoder_output")
    print(f"  3. Project to vocabulary → logits")
    print()

    # Run forward pass
    logits = model(src, tgt)

    print(f"Output logits shape: {logits.shape}")
    print(f"  (batch={batch_size}, tgt_seq_len={tgt_seq_len}, vocab_size={tgt_vocab_size})")
    print()

    # Show probability distribution for first token of first sequence
    probs = torch.softmax(logits[0, 0, :], dim=-1)
    top_tokens = probs.topk(5)

    print(f"Output probabilities for position 0, sequence 0:")
    for rank, (idx, prob) in enumerate(zip(top_tokens.indices, top_tokens.values)):
        print(f"  Token {idx.item():5d}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    print()


def demonstrate_shapes():
    """Show shapes at each step of the Transformer."""
    print("=" * 70)
    print("SHAPE TRACKING THROUGH THE TRANSFORMER")
    print("=" * 70)
    print()

    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 4
    d_model = 16

    print(f"Config: batch={batch_size}, src_seq={src_seq_len}, tgt_seq={tgt_seq_len}, d_model={d_model}")
    print()

    shapes = [
        ("Step", "Operation", "Shape"),
        ("─" * 5, "─" * 35, "─" * 40),
        ("1", "Source input tokens", "(batch, src_seq_len)"),
        ("2", "Source embedding", "(batch, src_seq_len, d_model)"),
        ("3", "Source + positional encoding", "(batch, src_seq_len, d_model)"),
        ("4", "After encoder layers", "(batch, src_seq_len, d_model)"),
        ("5", "Target input tokens", "(batch, tgt_seq_len)"),
        ("6", "Target embedding", "(batch, tgt_seq_len, d_model)"),
        ("7", "Target + positional encoding", "(batch, tgt_seq_len, d_model)"),
        ("8", "After decoder layers", "(batch, tgt_seq_len, d_model)"),
        ("9", "After output linear", "(batch, tgt_seq_len, vocab_size)"),
        ("10", "After softmax", "(batch, tgt_seq_len, vocab_size)"),
    ]

    for step, op, shape in shapes[1:]:
        print(f"  {step}. {op:<35} {shape:<40}")

    print()


def demonstrate_attention_flow():
    """Show how information flows through the model."""
    print("=" * 70)
    print("INFORMATION FLOW DIAGRAM")
    print("=" * 70)
    print()

    print("  SOURCE TEXT: \"The cat sat\"")
    print("  TARGET TEXT: \"Le chat s'est assis\"")
    print()

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    ENCODER PATH                             │")
    print("  │                                                             │")
    print('  │  "The" → Embedding → Positional → Encoder Layer 1          │')
    print('  │  "cat" → Embedding → Positional → Encoder Layer 1          │')
    print('  │  "sat" → Embedding → Positional → Encoder Layer 1          │')
    print("  │                              ↓                             │")
    print("  │                        Encoder Layer 2                     │")
    print("  │                              ↓                             │")
    print("  │                    Contextualized                          │")
    print("  │                    Representations                         │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print("                                        │")
    print("                                        ▼")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    DECODER PATH                             │")
    print("  │                                                             │")
    print('  │  "<sos>" → Embedding → Positional → Decoder Layer 1        │')
    print('  │    ↕ Cross-Attention ↕                                     │')
    print('  │    (looks at encoder output)                                │')
    print("  │                              ↓                             │")
    print("  │                        Decoder Layer 2                     │")
    print('  │    ↕ Cross-Attention ↕                                     │')
    print("  │                              ↓                             │")
    print('  │  "Le" → Embedding → Positional → Decoder Layer 1          │')
    print('  │    ↕ Cross-Attention ↕                                     │')
    print("  │                              ↓                             │")
    print("  │                        Decoder Layer 2                     │")
    print("  │                              ↓                             │")
    print('  │  Output: probabilities over entire vocab                   │')
    print('  │  → "Le" has highest probability ✓                          │')
    print("  └─────────────────────────────────────────────────────────────┘")
    print()


def show_layer_breakdown():
    """Show the breakdown of layers in the model."""
    print("=" * 70)
    print("MODEL LAYER BREAKDOWN")
    print("=" * 70)
    print()

    print("TRANSFORMER MODEL")
    print("├── Source Embedding (vocab × d_model)")
    print("├── Source Positional Encoding (sinusoidal)")
    print("├── Encoder")
    print("│   ├── Layer 1")
    print("│   │   ├── Multi-Head Self-Attention")
    print("│   │   │   ├── W_Q (d_model × d_model)")
    print("│   │   │   ├── W_K (d_model × d_model)")
    print("│   │   │   ├── W_V (d_model × d_model)")
    print("│   │   │   └── W_O (d_model × d_model)")
    print("│   │   ├── Feed-Forward (d_model → d_ff → d_model)")
    print("│   │   ├── LayerNorm × 2")
    print("│   │   └── Dropout × 2")
    print("│   ├── Layer 2")
    print("│   │   └── (same as Layer 1)")
    print("│   └── ...")
    print("├── Target Embedding (vocab × d_model)")
    print("├── Target Positional Encoding (sinusoidal)")
    print("├── Decoder")
    print("│   ├── Layer 1")
    print("│   │   ├── Masked Multi-Head Self-Attention")
    print("│   │   ├── Cross-Attention")
    print("│   │   │   ├── W_Q (d_model × d_model)")
    print("│   │   │   ├── W_K (d_model × d_model)")
    print("│   │   │   ├── W_V (d_model × d_model)")
    print("│   │   │   └── W_O (d_model × d_model)")
    print("│   │   ├── Feed-Forward (d_model → d_ff → d_model)")
    print("│   │   ├── LayerNorm × 3")
    print("│   │   └── Dropout × 3")
    print("│   ├── Layer 2")
    print("│   │   └── (same as Layer 1)")
    print("│   └── ...")
    print("└── Output Linear (d_model × vocab)")
    print()


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "TRANSFORMER FROM SCRATCH - LESSON 6" + " " * 23 + "║")
    print("║" + " " * 18 + "Full Transformer Model" + " " * 30 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Demo 1: Full model
    demonstrate_full_transformer()
    print()

    # Demo 2: Shape tracking
    demonstrate_shapes()
    print()

    # Demo 3: Information flow
    demonstrate_attention_flow()
    print()

    # Demo 4: Layer breakdown
    show_layer_breakdown()
    print()

    print("=" * 70)
    print("SUMMARY OF LESSON 6:")
    print("-" * 70)
    print("✓ Complete Transformer: Encoder + Decoder + Embeddings")
    print("✓ Masks: Source padding, Target causal, Target padding")
    print("✓ Forward pass: Encode → Decode → Project → Logits")
    print()
    print("MODEL SUMMARY:")
    print("  Encoder: Embed → PosEncode → [EncoderLayer × N]")
    print("  Decoder: Embed → PosEncode → [DecoderLayer × N] → Linear")
    print("  Connection: Cross-attention in decoder queries encoder output")
    print()
    print("NEXT: Train the Transformer on a tiny translation task!")
    print("Run: python 07_train_translate.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()