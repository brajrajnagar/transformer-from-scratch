"""
Transformer from Scratch - Lesson 3: Multi-Head Self-Attention
===============================================================

This is the CORE mechanism that makes the Transformer powerful.

WHAT WE'VE BUILT SO FAR:
  Raw Text → Token IDs → Embeddings → + Position Encoding → Input to Encoder

TODAY'S PLACE IN PIPELINE:
  Input embeddings → SELF-ATTENTION → Contextualized representations

WHAT WE'LL BUILD:
  1. Scaled Dot-Product Attention (the fundamental operation)
  2. Multi-Head Attention (multiple attention "perspectives")
"""

import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    The fundamental attention operation.

    COMPUTE:
    ─────────
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    WHERE:
    ──────
    Q (Query):    What am I looking for?  shape: (batch, n_heads, seq_q, d_k)
    K (Key):      What do I contain?      shape: (batch, n_heads, seq_k, d_k)
    V (Value):    What info do I carry?   shape: (batch, n_heads, seq_k, d_v)

    STEPS:
    ──────
    1. Compute similarity: scores = Q @ K^T / sqrt(d_k)
       shape: (batch, n_heads, seq_q, seq_k)
    2. Apply softmax: attention_weights = softmax(scores, dim=-1)
       Each row sums to 1.0 → these are "attention weights"
    3. Weighted sum: output = attention_weights @ V
       shape: (batch, n_heads, seq_q, d_v)

    WHY sqrt(d_k)?
    ──────────────
    When d_k is large, the dot products Q@K^T can have large variance.
    Large values → softmax becomes very peaked (near one-hot) → gradients
    vanish. Dividing by sqrt(d_k) keeps values in a good range.

    This is why it's called "SCALE"d dot-product attention.
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: Query tensor, shape (batch, seq_q, d_model)
            K: Key tensor, shape (batch, seq_k, d_model)
            V: Value tensor, shape (batch, seq_k, d_model)
            mask: Optional mask tensor (for decoder padding/causality)

        Returns:
            output: Attention output, shape (batch, seq_q, d_model)
            attention_weights: Softmax weights, shape (batch, seq_q, seq_k)
        """
        batch_size = Q.shape[0]
        d_k = Q.shape[-1]

        # Step 1: Compute raw attention scores
        # Q: (batch, seq_q, d_k), K^T: (d_k, seq_k) → scores: (batch, seq_q, seq_k)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)

        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Step 2: Softmax → attention weights
        # Each row sums to 1.0
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 3: Weighted sum of values
        # weights: (batch, seq_q, seq_k), V: (batch, seq_k, d_k) → output: (batch, seq_q, d_k)
        output = torch.bmm(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: Run attention multiple times in parallel.

    WHY MULTI-HEAD?
    ───────────────
    Think of each "head" as having different glasses:
      Head 1  → sees subject-verb relationships
      Head 2  → sees pronoun-antecedent relationships
      Head 3  → sees adjective-noun relationships
      Head 4  → sees temporal/causal relationships

    Each head learns DIFFERENT attention patterns. Then we combine them.

    THE PROCESS:
    ────────────
    1. Project Q, K, V into lower-dimensional spaces (d_model → n_heads × d_k)
    2. Split into multiple heads (each head operates in d_k dimensions)
    3. Apply scaled dot-product attention to EACH head
    4. Concatenate all heads
    5. Project back to d_model

    DIMENTIONS:
    ───────────
    Input:  (batch, seq, d_model)
    Where:  d_model = n_heads × d_k

    Step 1: Linear projection → (batch, seq, d_model)  [3 copies: Q, K, V]
    Step 2: Reshape → (batch, seq, n_heads, d_k)
    Step 3: Transpose → (batch, n_heads, seq, d_k)
    Step 4: Attention → (batch, n_heads, seq, d_k)
    Step 5: Transpose → (batch, seq, n_heads, d_k)
    Step 6: Reshape → (batch, seq, n_heads × d_k) = (batch, seq, d_model)
    Step 7: Linear project → (batch, seq, d_model)
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension of each head

        # Linear projections for Q, K, V and the final output
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Dropout for training
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k) and transpose.

        Input:  (batch, seq_len, d_model)
        Output: (batch, n_heads, seq_len, d_k)
        """
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge the heads back together.

        Input:  (batch, n_heads, seq_len, d_k)
        Output: (batch, seq_len, d_model)
        """
        batch, n_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2)  # (batch, seq_len, n_heads, d_k)
        return x.contiguous().view(batch, seq_len, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: Query, shape (batch, seq_q, d_model)
            K: Key, shape (batch, seq_k, d_model)
            V: Value, shape (batch, seq_k, d_model)
            mask: Optional mask (batch, 1, 1, seq_k) or (batch, 1, seq_k)

        Returns:
            output: shape (batch, seq_q, d_model)
            attention_weights: shape (batch, n_heads, seq_q, seq_k)
        """
        batch_size = Q.shape[0]

        # Step 1: Linear projections
        Q = self.W_Q(Q)  # (batch, seq_q, d_model)
        K = self.W_K(K)  # (batch, seq_k, d_model)
        V = self.W_V(V)  # (batch, seq_k, d_model)

        # Step 2: Split into multiple heads
        Q = self._split_heads(Q)  # (batch, n_heads, seq_q, d_k)
        K = self._split_heads(K)  # (batch, n_heads, seq_k, d_k)
        V = self._split_heads(V)  # (batch, n_heads, seq_k, d_v)

        # Step 3: Scaled dot-product attention (per head)
        # We can use a single matrix multiply for all heads at once
        # scores: (batch, n_heads, seq_q, seq_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, 1, seq_k)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Step 4: Softmax → attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch, n_heads, seq_q, seq_k)
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: Apply attention weights to values
        # output_per_head: (batch, n_heads, seq_q, d_k)
        output_per_head = torch.matmul(attention_weights, V)

        # Step 6: Concatenate heads
        output = self._merge_heads(output_per_head)  # (batch, seq_q, d_model)

        # Step 7: Final linear projection
        output = self.W_O(output)  # (batch, seq_q, d_model)

        output = self.output_dropout(output)

        return output, attention_weights


def demonstrate_scaled_dot_product_attention():
    """Show how scaled dot-product attention works step by step."""
    print("=" * 70)
    print("SCALED DOT-PRODUCT ATTENTION DEMO")
    print("=" * 70)
    print()

    batch_size = 1
    seq_len = 4
    d_k = 8

    # Create simple Q, K, V
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    print(f"Input shapes:")
    print(f"  Q (queries):  {Q.shape}")
    print(f"  K (keys):     {K.shape}")
    print(f"  V (values):   {V.shape}")
    print()

    # Step 1: Compute scores
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    print(f"Step 1: Raw attention scores = Q @ K^T / sqrt(d_k)")
    print(f"  Shape: {scores.shape}")
    print(f"  Values:\n{scores[0]}")
    print()

    # Step 2: Softmax
    attention_weights = torch.softmax(scores, dim=-1)
    print(f"Step 2: Softmax → attention weights")
    print(f"  Shape: {attention_weights.shape}")
    print(f"  Values:\n{attention_weights[0]}")
    print(f"  Row sums: {attention_weights[0].sum(dim=-1).tolist()}")
    print()

    # Step 3: Weighted sum
    output = attention_weights @ V
    print(f"Step 3: Weighted sum = weights @ V")
    print(f"  Shape: {output.shape}")
    print()

    # Interpretation
    print("INTERPRETATION:")
    print("-" * 70)
    words = ["[CLS]", "the", "cat", "sat"]
    print(f"  Words: {words}")
    print()
    for i, word in enumerate(words):
        print(f"  '{word}' attends to:")
        weights_i = attention_weights[0, i].tolist()
        for j, (w, wt) in enumerate(zip(words, weights_i)):
            bar = "█" * int(wt * 40)
            print(f"    → {w:8s}: {wt:.3f} {bar}")
    print()


def demonstrate_multi_head_attention():
    """Show how multi-head attention works."""
    print("=" * 70)
    print("MULTI-HEAD ATTENTION DEMO")
    print("=" * 70)
    print()

    d_model = 64
    n_heads = 4
    d_k = d_model // n_heads

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)

    batch_size = 2
    seq_len_q = 5  # query length (decoder)
    seq_len_k = 5  # key/value length (encoder)

    Q = torch.randn(batch_size, seq_len_q, d_model)
    K = torch.randn(batch_size, seq_len_k, d_model)
    V = torch.randn(batch_size, seq_len_k, d_model)

    print(f"Configuration:")
    print(f"  d_model = {d_model}")
    print(f"  n_heads = {n_heads}")
    print(f"  d_k = d_model / n_heads = {d_k}")
    print()

    print(f"Input shapes:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    print()

    output, attention_weights = mha(Q, K, V)

    print(f"Output shapes:")
    print(f"  output:           {output.shape}")
    print(f"  attention_weights:{attention_weights.shape}")
    print(f"    (batch, n_heads, seq_q, seq_k)")
    print()

    print(f"Attention weights for head 0:")
    print(f"  {attention_weights[0, 0].tolist()}")
    print()

    print(f"Each head learns DIFFERENT attention patterns:")
    for h in range(n_heads):
        weights = attention_weights[0, h]
        print(f"  Head {h}: max attention at position {weights.argmax(dim=-1).tolist()}")
    print()


def demonstrate_masked_attention():
    """Show causal masking for decoder (prevent looking ahead)."""
    print("=" * 70)
    print("MASKED ATTENTION (Decoder - No Looking Ahead!)")
    print("=" * 70)
    print()

    print("In the DECODER, we must prevent tokens from attending to")
    print("FUTURE positions. This is crucial for autoregressive generation.")
    print()

    seq_len = 5
    words = ["I", "love", "transformers"]

    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"Causal mask (lower triangular):")
    print(f"  1 = visible, 0 = masked")
    print(f"  {mask.tolist()}")
    print()

    print("  Rows = query (what I am)")
    print("  Cols = key (what I can attend to)")
    print()

    print("Interpretation:")
    print(f"  'I' (pos 0) can attend to: [I]")
    print(f"  'love' (pos 1) can attend to: [I, love]")
    print(f"  'transformers' (pos 2) can attend to: [I, love, transformers]")
    print()

    # Demo with actual attention
    d_model = 32
    n_heads = 4
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)

    Q = torch.randn(1, seq_len, d_model)
    K = torch.randn(1, seq_len, d_model)
    V = torch.randn(1, seq_len, d_model)

    output, attn = mha(Q, K, V, mask=mask)

    print("With mask applied, attention weights are:")
    for i in range(seq_len):
        weights = attn[0, 0, i].tolist()
        print(f"  Position {i}: {[f'{w:.3f}' for w in weights]}")
    print()


def demonstrate_attention_visualization():
    """Create a visual representation of attention patterns."""
    print("=" * 70)
    print("ATTENTION PATTERN VISUALIZATION")
    print("=" * 70)
    print()

    # Simple example with 6 tokens
    seq_len = 6
    d_model = 32
    n_heads = 4

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)

    # Create input
    x = torch.randn(1, seq_len, d_model)
    output, attn = mha(x, x, x)  # Self-attention: Q=K=V=x

    words = ["[BOS]", "the", "cat", "sat", "on", "mat"]

    print(f"Self-attention patterns for: {' '.join(words)}")
    print()

    for h in range(n_heads):
        print(f"Head {h}:")
        for i in range(seq_len):
            weights = attn[0, h, i].tolist()
            # Show which positions get most attention
            top_k = weights.index(max(weights))
            marker = "←" if i == top_k else "  "
            bar = "█" * int(weights[top_k] * 30)
            print(f"  {marker} {words[i]:8s} → {words[top_k]:8s} ({weights[top_k]:.2%}) {bar}")
        print()


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "TRANSFORMER FROM SCRATCH - LESSON 3" + " " * 26 + "║")
    print("║" + " " * 15 + "Multi-Head Self-Attention" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Demo 1: Scaled dot-product attention
    demonstrate_scaled_dot_product_attention()
    print()

    # Demo 2: Multi-head attention
    demonstrate_multi_head_attention()
    print()

    # Demo 3: Masked attention
    demonstrate_masked_attention()
    print()

    # Demo 4: Attention visualization
    demonstrate_attention_visualization()
    print()

    print("=" * 70)
    print("SUMMARY OF LESSON 3:")
    print("-" * 70)
    print("✓ ScaledDotProductAttention: softmax(Q@K^T/sqrt(d_k)) @ V")
    print("✓ MultiHeadAttention: Multiple attention heads in parallel")
    print("✓ Each head learns different attention patterns")
    print("✓ Causal mask: Prevent decoder from looking ahead")
    print()
    print("Shapes:")
    print("  Input:  (batch, seq, d_model)")
    print("  Output: (batch, seq, d_model)")
    print()
    print("KEY FORMULAS:")
    print("  scores = Q @ K^T / sqrt(d_k)")
    print("  attention = softmax(scores) @ V")
    print("  multi_head = concat(head_1, ..., head_n) @ W_O")
    print()
    print("NEXT: Encoder Layer (attention + feed-forward + residual + layernorm)")
    print("Run: python 04_encoder_layer.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()