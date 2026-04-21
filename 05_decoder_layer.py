"""
Transformer from Scratch - Lesson 5: Decoder Layer
===================================================

The decoder is the MOST COMPLEX component. It has THREE sublayers.

WHAT WE'VE BUILT SO FAR:
  1. Token + Position Embeddings
  2. Multi-Head Self-Attention
  3. Encoder Layer (attention + FFN + residual + layernorm)

TODAY'S PLACE IN PIPELINE:
  Encoder Output + Target Text → [Decoder Layer × N] → Decoder Output

WHAT WE'LL BUILD:
  1. Masked Multi-Head Self-Attention (prevent looking ahead)
  2. Multi-Head Cross-Attention (attend to encoder output)
  3. Feed-Forward Network (same as encoder)
  4. Residual + LayerNorm (same as encoder)
  5. Complete Decoder Layer
  6. Stack of Decoder Layers
"""

import math
import torch
import torch.nn as nn


# ============================================================================
# COMPONENT 1: Masked Multi-Head Self-Attention
# ============================================================================

class MaskedMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with CAUSAL MASKING.

    WHY CAUSAL MASKING?
    ──────────────────
    During training, we feed the ENTIRE target sequence at once:
      "Je suis" (French for "I am")

    But during inference, we generate ONE token at a time:
      Step 1: "Je"
      Step 2: "Je suis"
      Step 3: "Je suis ..."

    To make training match inference behavior, we MASK the decoder so
    each position can ONLY attend to PREVIOUS positions (and itself).

    This is called "causal" attention because position t can only depend
    on positions ≤ t.

    THE MASK:
    ─────────
    For a sequence of length 5:
      [[1, 0, 0, 0, 0],    Position 0 sees: only itself
       [1, 1, 0, 0, 0],    Position 1 sees: 0, 1
       [1, 1, 1, 0, 0],    Position 2 sees: 0, 1, 2
       [1, 1, 1, 1, 0],    Position 3 sees: 0, 1, 2, 3
       [1, 1, 1, 1, 1]]    Position 4 sees: 0, 1, 2, 3, 4

    This is a LOWER TRIANGULAR matrix.
    """

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Causal mask (lower triangular), shape (seq_len, seq_len)
                  or (batch, 1, seq_len, seq_len)
        Returns:
            output: shape (batch, seq_len, d_model)
            attention_weights: shape (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self._split_heads(self.W_Q(x))  # (batch, n_heads, seq, d_k)
        K = self._split_heads(self.W_K(x))  # (batch, n_heads, seq, d_k)
        V = self._split_heads(self.W_V(x))  # (batch, n_heads, seq, d_k)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, n_heads, seq, seq)

        # Apply causal mask (set masked positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax + dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply attention to values
        output = self._merge_heads(torch.matmul(attn_weights, V))
        output = self.W_O(output)
        output = self.output_dropout(output)

        return output, attn_weights


# ============================================================================
# COMPONENT 2: Multi-Head Cross-Attention
# ============================================================================

class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention: Decoder attends to Encoder output.

    HOW IT WORKS:
    ─────────────
    In cross-attention, the roles are different:
      Q comes from the DECODER (what am I looking for?)
      K and V come from the ENCODER (what does the input contain?)

    This allows each decoder position to LOOK AT the entire encoder output
    and pick out relevant information.

    EXAMPLE (English → French translation):
    ──────────────────────────────────────
    Encoder input:  "The cat sat"
    Decoder output (so far): "Je"

    Cross-attention:
      Q = embedding of "Je"     (what am I looking for?)
      K, V = encoder output     (what info is available?)

    The decoder learns: "To translate 'Je', I should pay attention to
    'cat' in the English input."

    SHAPES:
    ───────
    Q:     (batch, seq_decoder, d_model)
    K, V:  (batch, seq_encoder, d_model)
    Output: (batch, seq_decoder, d_model)

    NOTE: seq_decoder and seq_encoder can be DIFFERENT lengths!
    """

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

    def forward(
        self,
        decoder_input: torch.Tensor,  # Q
        encoder_output: torch.Tensor,  # K, V
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_input: Decoder embeddings, shape (batch, seq_dec, d_model)
            encoder_output: Encoder output, shape (batch, seq_enc, d_model)
            mask: Optional mask
        Returns:
            output: shape (batch, seq_dec, d_model)
            attention_weights: shape (batch, n_heads, seq_dec, seq_enc)
        """
        batch_size = decoder_input.shape[0]

        # Q from decoder, K and V from encoder
        Q = self._split_heads(self.W_Q(decoder_input))   # (batch, n_heads, seq_dec, d_k)
        K = self._split_heads(self.W_K(encoder_output))   # (batch, n_heads, seq_enc, d_k)
        V = self._split_heads(self.W_V(encoder_output))   # (batch, n_heads, seq_enc, d_k)

        # Compute attention scores
        # (batch, n_heads, seq_dec, seq_enc)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        output = self._merge_heads(torch.matmul(attn_weights, V))
        output = self.W_O(output)
        output = self.output_dropout(output)

        return output, attn_weights


# ============================================================================
# COMPONENT 3: Feed-Forward Network (same as encoder)
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
# COMPONENT 4: Layer Normalization (same as encoder)
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
# MAIN: Decoder Layer
# ============================================================================

class DecoderLayer(nn.Module):
    """
    A single Transformer Decoder Layer.

    ARCHITECTURE (THREE SUBLAYERS):
    ───────────────────────────────
    Input: (batch, seq_dec, d_model)
      ↓
    ┌─────────────────────────────────────────────────────┐
    │ Sublayer 1: Masked Multi-Head Self-Attention        │
    │   - Decoder attends to its own PREVIOUS positions   │
    │   - Q = K = V = decoder input                       │
    └─────────────────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────────────────┐
    │ Residual + LayerNorm                                │
    └─────────────────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────────────────┐
    │ Sublayer 2: Multi-Head Cross-Attention              │
    │   - Decoder attends to ENCODER output               │
    │   - Q = decoder input, K,V = encoder output         │
    └─────────────────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────────────────┐
    │ Residual + LayerNorm                                │
    └─────────────────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────────────────┐
    │ Sublayer 3: Feed-Forward Network                    │
    │   - Position-wise, same as encoder                  │
    └─────────────────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────────────────┐
    │ Residual + LayerNorm                                │
    └─────────────────────────────────────────────────────┘
          ↓
    Output: (batch, seq_dec, d_model)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Sublayer 1: Masked self-attention (decoder only looks backward)
        self.masked_attention = MaskedMultiHeadAttention(d_model, n_heads, dropout)

        # Sublayer 2: Cross-attention (decoder looks at encoder output)
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)

        # Sublayer 3: Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

        # Layer normalization for each sublayer
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))
        self.norm3 = LayerNorm((d_model,))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,              # Decoder input
        encoder_output: torch.Tensor,  # From encoder
        causal_mask: torch.Tensor,     # Lower triangular mask
        padding_mask: torch.Tensor = None,  # For padding tokens
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input embeddings, shape (batch, seq_dec, d_model)
            encoder_output: Encoder output, shape (batch, seq_enc, d_model)
            causal_mask: Causal mask for self-attention
            padding_mask: Mask for padding tokens in encoder output
        Returns:
            Output, shape (batch, seq_dec, d_model)
        """
        batch_size, seq_dec, d_model = x.shape

        # ---- Sublayer 1: Masked Self-Attention ----
        # Decoder attends only to previous positions
        attn_output, _ = self.masked_attention(x, causal_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # ---- Sublayer 2: Cross-Attention ----
        # Decoder attends to encoder output
        cross_attn_output, _ = self.cross_attention(x, encoder_output, padding_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # ---- Sublayer 3: Feed-Forward Network ----
        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask.

    Returns a (seq_len, seq_len) tensor where:
      1 = can attend
      0 = cannot attend (future position)
    """
    return torch.tril(torch.ones(seq_len, seq_len))


def demonstrate_causal_mask():
    """Show how causal masking works."""
    print("=" * 70)
    print("CAUSAL MASK DEMONSTRATION")
    print("=" * 70)
    print()

    seq_len = 5
    mask = create_causal_mask(seq_len)

    print(f"Causal mask for seq_len={seq_len}:")
    print(f"  Shape: {mask.shape}")
    print()
    print("  Columns (what I can attend to):")
    print("       0    1    2    3    4")
    for i in range(seq_len):
        row = " ".join([f"{mask[i,j]:.0f}" for j in range(seq_len)])
        print(f"  Row {i}:  {row}")
    print()
    print("  Rows = what I am")
    print("  Cols = what I can see")
    print()

    words = ["Je", "suis", "un", "chat", "!"]
    print(f"Words: {words}")
    print()
    print("During training:")
    print(f"  'Je'    can only see: [Je]")
    print(f"  'suis'  can see: [Je, suis]")
    print(f"  'un'    can see: [Je, suis, un]")
    print(f"  'chat'  can see: [Je, suis, un, chat]")
    print(f"  '!'     can see: [Je, suis, un, chat, !]")
    print()
    print("During inference (autoregressive):")
    print(f"  Step 1: Generate 'Je'   → only 'Je' is available")
    print(f"  Step 2: Generate 'suis' → 'Je suis' is available")
    print(f"  Step 3: Generate 'un'   → 'Je suis un' is available")
    print(f"  ...and so on")
    print()


def demonstrate_cross_attention():
    """Show how cross-attention works."""
    print("=" * 70)
    print("CROSS-ATTENTION DEMONSTRATION")
    print("=" * 70)
    print()

    batch_size = 1
    seq_dec = 3  # Decoder sequence length
    seq_enc = 4  # Encoder sequence length
    d_model = 32

    cross_attn = CrossAttention(d_model, n_heads=4, dropout=0.0)

    # Decoder input (so far generated: "Je suis un")
    decoder_input = torch.randn(batch_size, seq_dec, d_model)

    # Encoder output (input: "I am a cat")
    encoder_output = torch.randn(batch_size, seq_enc, d_model)

    print(f"Encoder input: ['I', 'am', 'a', 'cat'] (seq_len={seq_enc})")
    print(f"Decoder input: ['Je', 'suis', 'un'] (seq_len={seq_dec})")
    print()

    output, attn_weights = cross_attn(decoder_input, encoder_output)

    print(f"Cross-attention output shape: {output.shape}")
    print(f"  (batch={batch_size}, seq_dec={seq_dec}, d_model={d_model})")
    print()

    print(f"Attention weights (decoder → encoder):")
    print(f"  Shape: {attn_weights.shape}")
    print(f"  (batch, n_heads, seq_dec, seq_enc)")
    print()

    encoder_words = ["I", "am", "a", "cat"]
    decoder_words = ["Je", "suis", "un"]

    print(f"  Decoder '{decoder_words[0]}' attends to encoder:")
    for h in range(4):
        weights = attn_weights[0, h, 0].tolist()
        print(f"    Head {h}: {dict(zip(encoder_words, [f'{w:.2f}' for w in weights]))}")

    print()
    print(f"  Decoder '{decoder_words[1]}' attends to encoder:")
    for h in range(4):
        weights = attn_weights[0, h, 1].tolist()
        print(f"    Head {h}: {dict(zip(encoder_words, [f'{w:.2f}' for w in weights]))}")
    print()


def demonstrate_decoder_layer():
    """Show a complete decoder layer in action."""
    print("=" * 70)
    print("DECODER LAYER DEMO")
    print("=" * 70)
    print()

    d_model = 64
    n_heads = 4
    d_ff = 128
    batch_size = 1
    seq_dec = 3
    seq_enc = 4

    decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout=0.0)

    # Create causal mask
    causal_mask = create_causal_mask(seq_dec)

    # Create dummy inputs
    decoder_input = torch.randn(batch_size, seq_dec, d_model)
    encoder_output = torch.randn(batch_size, seq_enc, d_model)

    print(f"Input shapes:")
    print(f"  Decoder input:    {decoder_input.shape}")
    print(f"  Encoder output:   {encoder_output.shape}")
    print()

    print("Processing through decoder layer:")
    print(f"  1. Masked Self-Attention: decoder looks backward only")
    print(f"  2. Residual + LayerNorm")
    print(f"  3. Cross-Attention: decoder looks at encoder output")
    print(f"  4. Residual + LayerNorm")
    print(f"  5. Feed-Forward: position-wise processing")
    print(f"  6. Residual + LayerNorm")
    print()

    output = decoder_layer(decoder_input, encoder_output, causal_mask)

    print(f"Output shape: {output.shape}")
    print(f"  (batch={batch_size}, seq_dec={seq_dec}, d_model={d_model})")
    print()


def demonstrate_full_decoder():
    """Show a stack of decoder layers."""
    print("=" * 70)
    print("FULL DECODER (Stack of N Layers)")
    print("=" * 70)
    print()

    d_model = 64
    n_heads = 4
    d_ff = 128
    n_layers = 3
    batch_size = 1
    seq_dec = 4
    seq_enc = 5

    decoder = nn.ModuleList([
        DecoderLayer(d_model, n_heads, d_ff, dropout=0.0)
        for _ in range(n_layers)
    ])

    causal_mask = create_causal_mask(seq_dec)

    decoder_input = torch.randn(batch_size, seq_dec, d_model)
    encoder_output = torch.randn(batch_size, seq_enc, d_model)

    print(f"Decoder: {n_layers} stacked layers")
    print(f"Decoder input shape:  {decoder_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print()

    for i, layer in enumerate(decoder):
        decoder_input = layer(decoder_input, encoder_output, causal_mask)
        print(f"After decoder layer {i+1}: {decoder_input.shape}")

    print()
    print(f"Final decoder output: {decoder_input.shape}")
    print()

    print("What each layer does:")
    print("  Layer 1: Combines self-attention + cross-attention from raw input")
    print("  Layer 2: Refines representations using deeper context")
    print("  Layer 3: More abstract reasoning over encoder + decoder info")
    print("  ...")
    print()


def compare_encoder_decoder():
    """Compare encoder and decoder layers side by side."""
    print("=" * 70)
    print("ENCODER vs DECODER COMPARISON")
    print("=" * 70)
    print()

    print(f"{'Component':<30} {'Encoder':<25} {'Decoder':<25}")
    print("-" * 80)
    print(f"{'Self-Attention':<30} {'Full (all→all)':<25} {'Masked (backward only)':<25}")
    print(f"{'Cross-Attention':<30} {'N/A':<25} {'Yes (to encoder)':<25}")
    print(f"{'Feed-Forward':<30} {'Yes':<25} {'Yes':<25}")
    print(f"{'Residual + LayerNorm':<30} {'2 times':<25} {'3 times':<25}")
    print(f"{'Input':<30} {'Embeddings':<25} {'Target Embeddings'}")
    print(f"{'Other input':<30} {'None':<25} {'Encoder output'}")
    print()

    print("KEY DIFFERENCE:")
    print("  Encoder: Understands the FULL input (bidirectional)")
    print("  Decoder: Generates one token at a time (causal, no looking ahead)")
    print()


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "TRANSFORMER FROM SCRATCH - LESSON 5" + " " * 20 + "║")
    print("║" + " " * 18 + "Decoder Layer" + " " * 36 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Demo 1: Causal mask
    demonstrate_causal_mask()
    print()

    # Demo 2: Cross-attention
    demonstrate_cross_attention()
    print()

    # Demo 3: Decoder layer
    demonstrate_decoder_layer()
    print()

    # Demo 4: Full decoder
    demonstrate_full_decoder()
    print()

    # Compare encoder vs decoder
    compare_encoder_decoder()
    print()

    print("=" * 70)
    print("SUMMARY OF LESSON 5:")
    print("-" * 70)
    print("✓ Masked Multi-Head Attention: causal mask prevents looking ahead")
    print("✓ Cross-Attention: decoder queries encoder output")
    print("✓ Feed-Forward: same as encoder")
    print("✓ Decoder Layer: MaskedAttn → Residual+LN → CrossAttn → Residual+LN → FFN → Residual+LN")
    print()
    print("DECODER LAYER FORMULA:")
    print("  x → LayerNorm(x + MaskedSelfAttn(x))")
    print("  → LayerNorm(x + CrossAttn(x, encoder_output))")
    print("  → LayerNorm(x + FFN(x))")
    print()
    print("NEXT: Combine everything into the FULL Transformer model!")
    print("Run: python 06_full_transformer.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()