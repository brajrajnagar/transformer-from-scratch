"""
Transformer from Scratch - Lesson 2: Embeddings & Position Encoding
====================================================================

In this lesson we implement the FIRST component of the Transformer pipeline:
converting raw text tokens into rich, position-aware embeddings.

PIPELINE SO FAR:
  Raw Text → Token IDs → Dense Embeddings → + Position Encoding → Encoder Input

WHAT WE'LL BUILD:
  1. Token Embedding: Look-up table that maps token IDs → dense vectors
  2. Positional Encoding: Sinusoidal signals that tell the model "where" each
     token appears in the sequence
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs (integers) into dense vectors (floats).

    This is essentially a look-up table:
      - Input:  tensor of token IDs, shape (batch_size, seq_len)
      - Output: tensor of embeddings, shape (batch_size, seq_len, d_model)

    Example:
      vocab_size = 10000   (10,000 unique words)
      d_model = 512        (each word becomes a 512-dimensional vector)

      token_ids = [[1, 5, 42]]   (batch of 1, sequence of 3 tokens)
      embedding = lookup[token_ids]  → shape (1, 3, 512)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        # nn.Embedding creates a look-up table:
        #   shape = (vocab_size, d_model)
        #   each row is the embedding for that token ID
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs, shape (batch_size, seq_len)

        Returns:
            Dense embeddings, shape (batch_size, seq_len, d_model)

        Note: Each embedding is scaled by sqrt(d_model).
        This was in the original paper and helps with training stability
        when combining with positional encodings.
        """
        # x shape: (batch_size, seq_len)
        # embedding lookup: each ID → one row of shape (d_model,)
        # Result: (batch_size, seq_len, d_model)
        # Then scale by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to token embeddings.

    WHY DO WE NEED THIS?
    ────────────────────
    Self-attention is PERMUTATION INVARIANT. If you shuffle the input tokens,
    the attention mechanism produces the same output (just shuffled). But the
    ORDER of words matters enormously in language!

    Solution: ADD position information to each token's embedding.

    THE FORMULA (from the original paper):
    ──────────────────────────────────────
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
      pos = position in the sequence (0, 1, 2, ..., seq_len-1)
      i   = dimension index (0, 1, 2, ..., d_model/2 - 1)
      d_model = embedding dimension (e.g., 512)

    WHY SINUSOIDAL?
    ───────────────
    1. Allows the model to attend to RELATIVE positions:
       For any fixed offset k, PE(pos+k) can be represented as a linear
       function of PE(pos). This is because sin(x+a) and cos(x+a) are
       linear combinations of sin(x) and cos(x).

    2. Works at ANY sequence length:
       The model can handle sequences longer than what it saw during training,
       because sinusoids are defined for all real numbers.

    3. Each position gets a UNIQUE encoding:
       No two positions have the same positional vector.

    ALTERNATIVE: Learned positional embeddings (used in many models)
    would just be a look-up table. Sinusoidal was chosen in the original
    Transformer for the reasons above.
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension (must be even)
            max_seq_len: Maximum sequence length the model will handle
            dropout: Dropout rate for regularization
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal position encoding"
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix: (max_seq_len, d_model)
        # All zeros initially
        pe = torch.zeros(max_seq_len, d_model)

        # Position vector: [0, 1, 2, ..., max_seq_len-1], shape (max_seq_len,)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Divisor: 10000^(-2i/d_model) for each dimension pair
        # Shape: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # Apply the formula:
        # Even dimensions: sin(pos * div_term)
        # Odd dimensions:  cos(pos * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices: 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices:  1, 3, 5, ...

        # Shape: (max_seq_len, d_model)
        # Register as buffer (not a parameter, doesn't need gradient)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings, shape (batch_size, seq_len, d_model)

        Returns:
            Position-aware embeddings, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Add positional encoding for the first seq_len positions
        # pe shape: (1, max_seq_len, d_model)
        # x + pe[:, :seq_len, :] → broadcasts across batch
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


def demonstrate_token_embedding():
    """Show how token embeddings work with a concrete example."""
    print("=" * 70)
    print("TOKEN EMBEDDING DEMO")
    print("=" * 70)
    print()

    # ================================================================
    # STEP 0: REAL TEXT -> TOKEN IDS (The tokenization process)
    # ================================================================
    print("STEP 0: How Raw Text Becomes Token IDs")
    print("-" * 70)
    print()

    # Create a simple vocabulary (in practice, you'd use a real tokenizer)
    vocab = {
        "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
        "the": 4, "a": 5, "cat": 6, "dog": 7, "sat": 8, "ran": 9,
        "on": 10, "mat": 11, "in": 12, "house": 13, "big": 14,
        "brown": 15, "jumped": 16, "over": 17, "lazy": 18,
        "and": 19, "played": 20, "happy": 21,
    }

    print("VOCABULARY (word -> ID):")
    for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"    {word:12s} -> {idx}")
    print()

    # Tokenize sentences
    sentences = [
        "the cat sat",
        "the dog ran",
    ]

    def tokenize(sentence, vocab):
        """Convert a sentence to token IDs."""
        tokens = sentence.lower().split()
        token_ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
        return token_ids, tokens

    print("Original sentences:")
    for s in sentences:
        print(f"    '{s}'")
    print()

    print("After tokenization:")
    for sentence in sentences:
        ids, tokens = tokenize(sentence, vocab)
        print(f"    '{sentence}'")
        print(f"      Words: {tokens}")
        print(f"      IDs:   {ids}")
        mapping = [f"'{w}'->{vocab[w]}" for w in tokens]
        print(f"      Map:   {' '.join(mapping)}")
        print()

    # ================================================================
    # STEP 1: TOKEN EMBEDDING (the actual embedding demo)
    # ================================================================
    vocab_size = 100
    d_model = 16
    batch_size = 2
    seq_len = 5

    # Create embedding layer
    token_emb = TokenEmbedding(vocab_size, d_model)

    # Use the real token IDs from above
    token_ids = torch.tensor([
        [4, 6, 8, 1, 1],    # "the cat sat <UNK> <UNK>"
        [4, 7, 9, 1, 1],    # "the dog ran <UNK> <UNK>"
    ])

    print(f"STEP 1: Token Embedding (Look-up Table)")
    print(f"  Input: Token IDs")
    print(f"  Shape: {token_ids.shape}")
    print(f"  Values:\n{token_ids}")
    print()
    print(f"  Interpretation: Each number is a word index.")
    print(f"    4 = 'the', 6 = 'cat', 8 = 'sat'")
    print(f"    4 = 'the', 7 = 'dog', 9 = 'ran'")
    print()

    # Look up embeddings
    embeddings = token_emb(token_ids)

    print(f"Output: Dense Embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  (batch_size={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()
    print(f"  First token of first sentence embedding (first 8 values):")
    print(f"    {embeddings[0, 0, :8].tolist()}")
    print()
    print(f"  Why scaling by sqrt(d_model) = {math.sqrt(d_model):.4f}?")
    print(f"    This matches the scale of positional encodings.")
    print(f"    So when we ADD them, neither dominates the other.")
    print()


def demonstrate_positional_encoding():
    """Show how positional encoding works with a concrete example."""
    print("=" * 70)
    print("POSITIONAL ENCODING DEMO")
    print("=" * 70)
    print()

    d_model = 16
    max_seq_len = 20

    pos_enc = PositionalEncoding(d_model, max_seq_len, dropout=0.0)

    # Create dummy embeddings (all zeros, just to show positional encoding)
    batch_size = 1
    seq_len = 10
    dummy_embeddings = torch.zeros(batch_size, seq_len, d_model)

    # Add positional encoding
    encoded = pos_enc(dummy_embeddings)

    print(f"Positional encoding shapes:")
    print(f"  d_model = {d_model}")
    print(f"  max_seq_len = {max_seq_len}")
    print(f"  Input: (batch={batch_size}, seq={seq_len}, features={d_model})")
    print(f"  Output: (batch={batch_size}, seq={seq_len}, features={d_model})")
    print()

    print(f"Positional encoding values for first position (pos=0):")
    print(f"  PE[0, :] = {encoded[0, 0, :8].tolist()}")
    print()

    print(f"Positional encoding values for second position (pos=1):")
    print(f"  PE[1, :] = {encoded[0, 1, :8].tolist()}")
    print()

    print(f"Notice: Each position has a UNIQUE encoding!")
    print(f"  Position 0 first 4 values:  {encoded[0, 0, :4].tolist()}")
    print(f"  Position 1 first 4 values:  {encoded[0, 1, :4].tolist()}")
    print(f"  Position 5 first 4 values:  {encoded[0, 5, :4].tolist()}")
    print()

    # Show the pattern
    print(f"Full positional encoding matrix for first position (first 8 dims):")
    print(f"  {encoded[0, 0, :8].tolist()}")
    print()
    print(f"  Dimension 0: sin(0 / 10000^0)      = sin(0) = 0")
    print(f"  Dimension 1: cos(0 / 10000^0)      = cos(0) = 1")
    print(f"  Dimension 2: sin(0 / 10000^0.00625) ≈ sin(0) = 0")
    print(f"  Dimension 3: cos(0 / 10000^0.00625) ≈ cos(0) = 1")
    print()


def demonstrate_combined():
    """Show token + positional embedding combined."""
    print("=" * 70)
    print("COMBINED: TOKEN + POSITIONAL EMBEDDING")
    print("=" * 70)
    print()

    vocab_size = 100
    d_model = 16
    batch_size = 1
    seq_len = 5

    token_emb = TokenEmbedding(vocab_size, d_model)
    pos_enc = PositionalEncoding(d_model, max_seq_len=5000, dropout=0.0)

    # Token IDs
    token_ids = torch.tensor([[10, 20, 30, 40, 50]])

    # Step 1: Token embedding
    embedded = token_emb(token_ids)
    print(f"Step 1: Token Embedding")
    print(f"  Input token IDs: {token_ids.shape} → {token_ids}")
    print(f"  Output: {embedded.shape}")
    print()

    # Step 2: Add positional encoding
    final = pos_enc(embedded)
    print(f"Step 2: Add Positional Encoding")
    print(f"  Output: {final.shape}")
    print()

    # Step 3: Show what each row now represents
    print(f"What each row represents:")
    print(f"  Row 0: word '10' at position 0  → [token_features + pos_0]")
    print(f"  Row 1: word '20' at position 1  → [token_features + pos_1]")
    print(f"  Row 2: word '30' at position 2  → [token_features + pos_2]")
    print(f"  Row 3: word '40' at position 3  → [token_features + pos_3]")
    print(f"  Row 4: word '50' at position 4  → [token_features + pos_4]")
    print()

    print(f"Same word at different positions:")
    token_ids_same = torch.tensor([[5, 5, 5, 5, 5]])  # All same token
    embedded_same = token_emb(token_ids_same)
    final_same = pos_enc(embedded_same)
    print(f"  Token IDs: {token_ids_same.tolist()}")
    print(f"  Embedding[0, 0, :4] = {embedded_same[0, 0, :4].tolist()}  (pos 0)")
    print(f"  Embedding[0, 1, :4] = {embedded_same[0, 1, :4].tolist()}  (pos 1)")
    print(f"  Embedding[0, 2, :4] = {embedded_same[0, 2, :4].tolist()}  (pos 2)")
    print(f"  → Same token, DIFFERENT final embedding because of position!")
    print()


def verify_properties():
    """Verify key properties of positional encoding."""
    print("=" * 70)
    print("VERIFYING POSITIONAL ENCODING PROPERTIES")
    print("=" * 70)
    print()

    d_model = 64
    pos_enc = PositionalEncoding(d_model, max_seq_len=100, dropout=0.0)

    # Property 1: Each position is unique
    pe = pos_enc.pe  # (1, max_seq_len, d_model)
    pe = pe.squeeze(0)  # (max_seq_len, d_model)

    diffs = []
    for i in range(min(10, len(pe))):
        for j in range(i + 1, min(10, len(pe))):
            diff = (pe[i] - pe[j]).abs().sum().item()
            diffs.append(diff)

    print(f"Property 1: Each position has unique encoding")
    print(f"  Min L1 distance between any two positions (0-9): {min(diffs):.4f}")
    print(f"  ✓ All positions are distinct!" if min(diffs) > 0 else "  ✗ Problem!")
    print()

    # Property 2: Relative position encoding
    # PE(pos + k) should be a linear function of PE(pos)
    print(f"Property 2: Relative positions are encodable")
    print(f"  For any fixed offset k, PE(pos+k) = W_k @ PE(pos)")
    print(f"  This is because sin(x+a) and cos(x+a) are linear combinations")
    print(f"  of sin(x) and cos(x).")
    print(f"  ✓ Relative position info is built in!")
    print()

    # Property 3: Dimensions have different frequency
    print(f"Property 3: Different frequencies per dimension pair")
    print(f"  Lower dimensions → higher frequency (shorter period)")
    print(f"  Higher dimensions → lower frequency (longer period)")
    print(f"  This lets the model capture both local and global structure.")
    print()


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "TRANSFORMER FROM SCRATCH - LESSON 2" + " " * 26 + "║")
    print("║" + " " * 18 + "Embeddings + Position Encoding" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Demo 1: Token embeddings
    demonstrate_token_embedding()
    print()

    # Demo 2: Positional encoding
    demonstrate_positional_encoding()
    print()

    # Demo 3: Combined
    demonstrate_combined()
    print()

    # Verify properties
    verify_properties()
    print()

    print("=" * 70)
    print("SUMMARY OF LESSON 2:")
    print("-" * 70)
    print("✓ TokenEmbedding: Look-up table (ID → dense vector)")
    print("✓ PositionalEncoding: Sinusoidal signals (position → vector)")
    print("✓ Combined: embedding + positional_encoding = rich representation")
    print()
    print("Shapes:")
    print("  Input token IDs:      (batch_size, seq_len)")
    print("  Token embeddings:     (batch_size, seq_len, d_model)")
    print("  With position enc:    (batch_size, seq_len, d_model)")
    print()
    print("NEXT: Multi-Head Self-Attention (the heart of the Transformer!)")
    print("Run: python 03_multi_head_attention.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()