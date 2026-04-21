"""
Transformer Architecture Overview - Lesson 1
=============================================

This is the FIRST lesson in our series. We'll build the entire Transformer
architecture from scratch, component by component.

Today's Goal: Understand the BIG PICTURE of the Transformer architecture.

The Transformer (Vaswani et al., 2017) was a revolutionary architecture that
replaced RNNs/LSTMs with pure attention mechanisms for sequence processing.

KEY INSIGHT: Attention allows the model to look at ALL other tokens in the
sequence at once, making it highly parallelizable and great at capturing
long-range dependencies.

ARCHITECTURE OVERVIEW (from the original paper figure):
=======================================================

                    ┌─────────────────────────────────────────────────┐
                    │                  ENCODER                        │
                    │                                                 │
  Input Text ──────►│  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
                    │  │ Layer 1 │→│ Layer 2 │→│   ...   │        │
                    │  └─────────┘  └─────────┘  └─────────┘        │
                    │                                                 │
                    │  Each layer contains:                         │
                    │   ┌──────────────────┐                         │
                    │   │ Self-Attention   │                         │
                    │   │ + Feed-Forward   │                         │
                    │   └──────────────────┘                         │
                    └─────────────────────────────────────────────────┘
                                      │
                                      │ Contextualized
                                      │ representations
                                      ▼
                    ┌─────────────────────────────────────────────────┐
                    │                  DECODER                        │
                    │                                                 │
  Target Text ─────►│  ┌─────────┐  ┌─────────┐  ┌─────────┐   ┌────┴─────┐
                    │  │ Layer 1 │→│ Layer 2 │→│   ...   │→│ Linear │→Softmax
                    │  └─────────┘  └─────────┘  └─────────┘   └──────────┘
                    │                                                 │
                    │  Each layer contains:                         │
                    │   ┌──────────────────────────┐                 │
                    │   │ Masked Self-Attention    │                 │
                    │   │ + Cross-Attention   ─────┼──── Encoder     │
                    │   │ + Feed-Forward           │                 │
                    │   └──────────────────────────┘                 │
                    └─────────────────────────────────────────────────┘
                                      │
                                      ▼
                               Output Text

DATA FLOW (for translation: "Hello" → "Bonjour"):
==================================================

Step 1: ENCODER SIDE (understand input)
  "Hello" → Token IDs → Embedding + Position → Encoder Layers → Context vectors

Step 2: DECODER SIDE (generate output)
  "<BOS>" → Token IDs → Embedding + Position → Decoder Layers → Logits → "Bonjour"

Step 3: CROSS-ATTENTION connects them
  Decoder layers "read" encoder output via cross-attention layers.

Inside each ENCODER layer:
  Input → Self-Attention → Add+Norm → FFN → Add+Norm → Output

Inside each DECODER layer:
  Input → Masked Self-Attn → Add+Norm → Cross-Attn → Add+Norm → FFN → Add+Norm → Output

KEY HYPERPARAMETERS:
==================
"""

import torch
import torch.nn as nn
import math


class TransformerConfig:
    """Configuration for the Transformer architecture."""

    def __init__(
        self,
        vocab_size: int = 10000,    # Size of vocabulary
        d_model: int = 512,         # Dimension of embedding (the "width")
        n_heads: int = 8,           # Number of attention heads
        n_encoder_layers: int = 6,  # Number of encoder layers
        n_decoder_layers: int = 6,  # Number of decoder layers
        d_ff: int = 2048,           # Hidden dimension of feed-forward network
        d_output: int = 10000,      # Output vocabulary size
        dropout: float = 0.1,       # Dropout rate
        max_sequence_length: int = 204,  # Max sequence length
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_ff = d_ff
        self.d_output = d_output
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length

        # Sanity checks
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_v = d_model // n_heads  # Same as d_k in standard attention

    def __repr__(self):
        return (
            f"TransformerConfig(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  n_heads={self.n_heads},\n"
            f"  n_encoder_layers={self.n_encoder_layers},\n"
            f"  n_decoder_layers={self.n_decoder_layers},\n"
            f"  d_ff={self.d_ff},\n"
            f"  d_k={self.d_k},\n"
            f"  dropout={self.dropout},\n"
            f")"
        )


def print_architecture_diagram():
    """Print a visual diagram of the Transformer, matching the paper figure."""

    print("=" * 70)
    print("       TRANSFORMER ARCHITECTURE (from 'Attention Is All You Need')")
    print("=" * 70)
    print()

    print("                         ┌─────────────────────────────────────────┐")
    print("                         │              ENCODER                     │")
    print("                         │                                            │")
    print("   Input Text ──────────►│  ┌─────────┐  ┌─────────┐  ┌─────────┐   │")
    print("   (e.g., 'Hello')      │  │ Layer 1 │→│ Layer 2 │→│   ...   │   │")
    print("                        │  └─────────┘  └─────────┘  └─────────┘   │")
    print("                         │                                            │")
    print("                         │  Each encoder layer:                      │")
    print("                         │   ┌──────────────────────┐                │")
    print("                         │   │ Self-Attention       │                │")
    print("                         │   │ ──────────────────   │                │")
    print("                         │   │ Feed-Forward         │                │")
    print("                         │   └──────────────────────┘                │")
    print("                         └────────────────┬────────────────────────┘")
    print("                                          │")
    print("                                          │ Contextualized")
    print("                                          │ representations")
    print("                                          ▼")
    print("                         ┌─────────────────────────────────────────┐")
    print("                         │              DECODER                     │")
    print("                         │                                            │")
    print("   Target Text ─────────►│  ┌─────────┐  ┌─────────┐  ┌─────────┐   │")
    print("   (e.g., '<BOS>')       │  │ Layer 1 │→│ Layer 2 │→│   ...   │→│")
    print("                        │  └─────────┘  └─────────┘  └─────────┘   │")
    print("                         │                                            │")
    print("                         │  Each decoder layer:                      │")
    print("                         │   ┌──────────────────────────┐            │")
    print("                         │   │ Masked Self-Attention    │            │")
    print("                         │   │ ─────────────────────    │            │")
    print("                         │   │ Cross-Attention   ───────┼──── Encoder│")
    print("                         │   │ (queries encoder output) │            │")
    print("                         │   │ Feed-Forward             │            │")
    print("                         │   └──────────────────────────┘            │")
    print("                         └────────────────┬────────────────────────┘")
    print("                                          │")
    print("                                          ▼")
    print("                         ┌─────────────────────────────────────────┐")
    print("                         │  Linear → Softmax → Output Text         │")
    print("                         │  (e.g., 'Bonjour')                      │")
    print("                         └─────────────────────────────────────────┘")
    print()
    print()
    print("  KEY: Cross-attention arrows flow FROM encoder TO decoder")
    print("       Decoder queries (Q) attend to Encoder keys/values (K, V)")
    print()


def explain_attention_simple():
    """Explain the attention mechanism in simple terms."""

    print("=" * 70)
    print("               UNDERSTANDING ATTENTION (Simply!)")
    print("=" * 70)
    print()

    print("Imagine you're reading this sentence:")
    print('  "The cat sat on the mat because IT was tired."')
    print()
    print("When you reach 'IT', your brain looks back at 'cat' to understand")
    print("what 'IT' refers to. That's ATTENTION!")
    print()

    print("In the Transformer, EVERY token does this simultaneously:")
    print()
    print('  "The"  -> looks at: cat, sat, mat')
    print('  "cat"  -> looks at: The, sat, on, the, mat')
    print('  "sat"  -> looks at: cat, on, the, mat')
    print('  ...and so on for EVERY token')
    print()

    print("The math behind self-attention:")
    print("-" * 70)
    print("  For each token, we compute 3 vectors:")
    print("    Q (Query)    : What am I looking for?")
    print("    K (Key)      : What do I contain?")
    print("    V (Value)    : What information do I carry?")
    print()
    print("  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V")
    print()
    print("  Breakdown:")
    print("    1. Q @ K^T        : Compute similarity scores")
    print("    2. / sqrt(d_k)    : Scale to prevent large values")
    print("    3. softmax(…)     : Convert to probabilities (sum to 1)")
    print("    4. @ V            : Weighted sum of values")
    print()

    # Simple numerical example
    print("NUMERICAL EXAMPLE (simplified):")
    print("-" * 70)
    print()

    # Create simple example
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 2 queries, 2-dim
    K = torch.tensor([[1.0, 0.5], [0.5, 1.0]])   # 2 keys, 2-dim
    V = torch.tensor([[1.0, 2.0], [2.0, 1.0]])   # 2 values, 2-dim

    print(f"Q (queries):\n{Q}")
    print(f"\nK (keys):\n{K}")
    print(f"\nV (values):\n{V}")

    # Compute attention scores
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    print(f"\nRaw scores (Q @ K^T / sqrt(d_k)):\n{scores}")

    # Softmax
    attention_weights = torch.softmax(scores, dim=-1)
    print(f"\nAttention weights (softmax):\n{attention_weights}")
    print(f"(Each row sums to 1.0)")

    # Weighted sum
    output = attention_weights @ V
    print(f"\nOutput (weights @ V):\n{output}")
    print()


def explain_multi_head_attention():
    """Explain multi-head attention simply."""

    print("=" * 70)
    print("         MULTI-HEAD ATTENTION: Why 'Multiple Heads'?")
    print("=" * 70)
    print()

    print("Single head is like having ONE pair of eyes.")
    print("Multiple heads = multiple pairs of eyes, each seeing differently!")
    print()
    print("Example with 4 heads:")
    print()
    print("  Head 1  -> might learn: subject-verb relationships")
    print("  Head 2  -> might learn: pronoun-antecedent relationships")
    print("  Head 3  -> might learn: adjective-noun relationships")
    print("  Head 4  -> might learn: temporal relationships")
    print()
    print("Then we CONCATENATE all heads and project back to d_model.")
    print()

    # Visual diagram
    config = TransformerConfig()
    print(f"Configuration:")
    print(f"  d_model = {config.d_model}")
    print(f"  n_heads = {config.n_heads}")
    print(f"  d_k = d_v = d_model / n_heads = {config.d_k}")
    print()
    print("Each head operates in a lower-dimensional space:")
    print(f"  Each head: {config.d_k} dimensions -> attention -> {config.d_v} dimensions")
    print(f"  All heads concatenated: {config.n_heads} x {config.d_v} = {config.d_model}")
    print()


def explain_position_encoding():
    """Explain why position encoding is needed."""

    print("=" * 70)
    print("        POSITION ENCODING: Why do we need it?")
    print("=" * 70)
    print()

    print("Self-attention is PERMUTATION INVARIANT:")
    print("  'The cat sat'  -> same output as  'cat The sat'  -> 'sat The cat'")
    print()
    print("But word ORDER matters! So we ADD position information:")
    print()
    print("  final_embedding = token_embedding + position_embedding")
    print()

    print("The Transformer uses SINUSOIDAL position encodings:")
    print()
    print("  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))")
    print("  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
    print()
    print("Why sinusoids?")
    print("  1. Allows model to attend to relative positions (easy linear relation)")
    print("  2. Works at any sequence length (even longer than training)")
    print("  3. Each position gets a UNIQUE encoding")
    print()


def explain_residual_and_layernorm():
    """Explain residual connections and layer normalization."""

    print("=" * 70)
    print("   RESIDUAL CONNECTIONS + LAYER NORMALIZATION")
    print("=" * 70)
    print()

    print("RESIDUAL CONNECTION (Skip Connection):")
    print("-" * 40)
    print("  output = x + Sublayer(x)")
    print()
    print("  Why?")
    print("    - Helps gradients flow through deep networks")
    print("    - Easier to train (vanishing gradient problem)")
    print("    - Model can always choose to 'skip' the transformation")
    print()

    print("LAYER NORMALIZATION:")
    print("-" * 40)
    print("  Normalizes across the feature dimension (like BatchNorm, but")
    print("  normalizes per-sample instead of across batch)")
    print()
    print("  Why LayerNorm instead of BatchNorm?")
    print("    - Works well with small batch sizes")
    print("    - More stable for sequence models")
    print()

    print("COMBINED (after each sublayer):")
    print("  x -> Sublayer(x) -> + -> LayerNorm -> Output")
    print("       (residual connection)    ")
    print()


def main():
    """Main function to run the overview."""

    print()
    print("+" + "=" * 68 + "+")
    print("|" + " " * 15 + "TRANSFORMER FROM SCRATCH - LESSON 1" + " " * 23 + "|")
    print("|" + " " * 20 + "Architecture Overview" + " " * 28 + "|")
    print("+" + "=" * 68 + "+")
    print()

    # Print architecture diagram
    print_architecture_diagram()

    # Show config
    config = TransformerConfig()
    print("DEFAULT CONFIG (like the original paper):")
    print("-" * 70)
    print(config)
    print()

    # Explain attention
    explain_attention_simple()

    # Explain multi-head
    explain_multi_head_attention()

    # Explain position encoding
    explain_position_encoding()

    # Explain residual + layernorm
    explain_residual_and_layernorm()

    print("=" * 70)
    print("WHAT'S NEXT (Lesson 2):")
    print("-" * 70)
    print("We'll implement: EMBEDDINGS + POSITION ENCODING")
    print()
    print("This is the FIRST step of the pipeline:")
    print("  Input text -> Token IDs -> Dense embeddings -> + Position encoding")
    print()
    print("Run: python 02_embeddings.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()