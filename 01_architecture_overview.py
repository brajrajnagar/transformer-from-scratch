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

    print("In the Transformer, EVERY token looks at EVERY other token")
    print("and decides HOW MUCH to pay attention to each one.")
    print()
    print("Concrete example with sentence: 'The cat sat'")
    print()
    print("  Token 'The' wants to know: what noun am I modifying?")
    print("    -> pays HIGH attention to: 'cat'  (it's the noun!")
    print("    -> pays LOW attention to:  'sat'  (verb, not relevant)")
    print()
    print("  Token 'cat' wants to know: what's the subject of this sentence?")
    print("    -> pays HIGH attention to: 'The'  (it tells me singular/plural)")
    print("    -> pays MEDIUM attention to: 'sat' (it's what the cat did)")
    print()
    print("  Token 'sat' wants to know: who did this action?")
    print("    -> pays HIGH attention to: 'cat'  (the cat is sitting!)")
    print("    -> pays LOW attention to:  'The'  (article, not the actor)")
    print()

    # Show attention weights visually
    print("Here's what the ATTENTION WEIGHTS look like after training:")
    print("(How much each token pays attention to each other token)")
    print()
    print("                The    cat    sat")
    print("       The     [0.10  0.75   0.15]   <- 'cat' gets highest")
    print("       cat     [0.40  0.10   0.50]   <- 'sat' gets highest")
    print("       sat     [0.15  0.80   0.05]   <- 'cat' gets highest")
    print()
    print("Each ROW sums to 1.0 (it's a probability distribution).")
    print("Each token redistributes its 'attention budget' across all tokens.")
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


def explain_layers_vs_heads():
    """Explain the difference between layers and heads clearly."""

    print()
    print("=" * 70)
    print("     LAYERS vs HEADS: What's the Difference?")
    print("=" * 70)
    print()

    print("The Transformer has TWO dimensions. They mean VERY different things:")
    print()

    # --- HEADS ---
    print("1. ATTENTION HEADS = Multiple Perspectives (Width)")
    print("-" * 70)
    print()
    print("Imagine 8 different EXPERTS reading the same sentence together:")
    print()
    print("  Head 1  → Grammar expert: 'This noun needs a verb'")
    print("  Head 2  → Pronoun expert: 'IT refers to the cat'")
    print("  Head 3  → Location expert: 'ON the mat tells us where'")
    print("  Head 4  → Sentiment expert: 'TIRED tells us the mood'")
    print("  Head 5  → Temporal expert: 'SAT happened in the past'")
    print("  Head 6  → Entity expert: 'The cat' is one entity")
    print("  Head 7  → Preposition expert: 'ON' connects cat to mat'")
    print("  Head 8  → Overall expert: 'This is a simple statement'")
    print()
    print("ALL 8 heads work SIMULTANEOUSLY on the SAME layer.")
    print("They each see the same words but focus on DIFFERENT relationships.")
    print()

    print("VISUAL: 8 heads in ONE encoder layer")
    print()
    print("  Input: 'The cat sat on the mat'")
    print()
    print("  +-----------------------------------------+")
    print("  |       SINGLE ENCODER LAYER              |")
    print("  |                                         |")
    print("  |  Word:  The   cat   sat   on   the   mat|")
    print("  |           |     |     |     |    |     | |")
    print("  |  +-----------------+-----------------+  |")
    print("  |  |  Head 1: Grammar relationships    |  |")
    print("  |  |  Head 2: Pronoun resolution       |  |")
    print("  |  |  Head 3: Spatial relationships    |  |")
    print("  |  |  Head 4: Sentiment/emotion        |  |")
    print("  |  |  Head 5: Temporal info            |  |")
    print("  |  |  Head 6: Entity tracking          |  |")
    print("  |  |  Head 7: Prepositional phrases    |  |")
    print("  |  |  Head 8: Global context           |  |")
    print("  |  +-----------------+-----------------+  |")
    print("  |           | (concatenate all 8)         |")
    print("  |  Output: Rich representation of each word")
    print("  +-----------------------------------------+")
    print()

    # --- LAYERS ---
    print()
    print("2. ENCODER LAYERS = Processing Stages (Depth)")
    print("-" * 70)
    print()
    print("Imagine reading the SAME sentence MULTIPLE TIMES, getting deeper")
    print("understanding each time:")
    print()
    print("  Layer 1 → 'I see the words and their basic relationships'")
    print("  Layer 2 → 'I understand the phrases and structure'")
    print("  Layer 3 → 'I get the context and meaning'")
    print("  Layer 4 → 'I understand the intent and sentiment'")
    print("  Layer 5 → 'I can relate this to everything I know'")
    print("  Layer 6 → 'I have a deep, rich understanding'")
    print()
    print("ALL 6 layers process the input SEQUENTIALLY (one after another).")
    print("Each layer builds on the previous one.")
    print()

    print("VISUAL: 6 encoder layers processing sequentially")
    print()
    print("  Input: 'The cat sat on the mat'")
    print()
    print("  +------------------+")
    print("  |  Layer 1         |  <- Basic word relationships")
    print("  |  (shallow)       |     'cat' is near 'sat' and 'the'")
    print("  +------------------+")
    print("           |")
    print("           v")
    print("  +------------------+")
    print("  |  Layer 2         |  <- Phrase structure")
    print("  |                  |     'the cat' is a noun phrase")
    print("  +------------------+")
    print("           |")
    print("           v")
    print("  +------------------+")
    print("  |  Layer 3         |  <- Sentence meaning")
    print("  |                  |     'cat sat on mat' describes an event")
    print("  +------------------+")
    print("           |")
    print("           v")
    print("  +------------------+")
    print("  |  Layer 4         |  <- Context")
    print("  |                  |     The cat is subject, mat is location")
    print("  +------------------+")
    print("           |")
    print("           v")
    print("  +------------------+")
    print("  |  Layer 5         |  <- Deeper context")
    print("  |                  |     This is a simple factual statement")
    print("  +------------------+")
    print("           |")
    print("           v")
    print("  +------------------+")
    print("  |  Layer 6         |  <- Deep understanding")
    print("  |  (deepest)       |     Full contextualized representation")
    print("  +------------------+")
    print("           |")
    print("           v")
    print("  Output: Rich contextualized embeddings")
    print()

    # --- COMBINED ---
    print()
    print("3. COMBINED: Layers x Heads = Total Model Capacity")
    print("-" * 70)
    print()
    print("The full Transformer has BOTH dimensions:")
    print()
    print("  - 6 encoder layers (depth)")
    print("  - 8 attention heads per layer (width)")
    print()
    print("This means:")
    print("  - Each layer uses 8 different perspectives")
    print("  - There are 6 such layers, processing sequentially")
    print("  - Total attention computations: 6 layers x 8 heads = 48")
    print()

    print("FULL PICTURE: 6 layers, each with 8 heads")
    print()
    print("  Input")
    print("    |")
    print("    v")
    print("  +-----------------------------------------------+")
    print("  |  ENCODER LAYER 1                              |")
    print("  |  +-----------------------------------------+  |")
    print("  |  | 8 heads: Grammar, Pronouns, Spatial...  |  |")
    print("  |  +-----------------------------------------+  |")
    print("  +-----------------------------------------------+")
    print("    |")
    print("    v")
    print("  +-----------------------------------------------+")
    print("  |  ENCODER LAYER 2                              |")
    print("  |  +-----------------------------------------+  |")
    print("  |  | 8 heads: New perspectives from layer 1  |  |")
    print("  |  +-----------------------------------------+  |")
    print("  +-----------------------------------------------+")
    print("    |")
    print("    v")
    print("  +-----------------------------------------------+")
    print("  |  ... (layers 3, 4, 5) ...                     |")
    print("  +-----------------------------------------------+")
    print("    |")
    print("    v")
    print("  +-----------------------------------------------+")
    print("  |  ENCODER LAYER 6                              |")
    print("  |  +-----------------------------------------+  |")
    print("  |  | 8 heads: Deep, abstract representations |  |")
    print("  |  +-----------------------------------------+  |")
    print("  +-----------------------------------------------+")
    print("    |")
    print("    v")
    print("  Output (deep contextualized)")
    print()

    # --- ANALOGY ---
    print()
    print("4. ANALOGY: A Restaurant Kitchen")
    print("-" * 70)
    print()
    print("Think of making a complex dish:")
    print()
    print("  HEADS = Different chefs working simultaneously")
    print("    - Chef 1: Handles proteins")
    print("    - Chef 2: Handles vegetables")
    print("    - Chef 3: Handles sauces")
    print("    - ... (8 chefs total, all working at once)")
    print()
    print("  LAYERS = Different stages of cooking")
    print("    - Stage 1: Prep (wash, chop)")
    print("    - Stage 2: Cook (sear, saute)")
    print("    - Stage 3: Combine (mix ingredients)")
    print("    - Stage 4: Season (add spices)")
    print("    - Stage 5: Plate (beautiful presentation)")
    print("    - Stage 6: Final check (taste test)")
    print()
    print("  Total work: 6 stages x 8 chefs = 48 chef-stages of work!")
    print()

    # --- SUMMARY TABLE ---
    print()
    print("5. SUMMARY TABLE")
    print("-" * 70)
    print()
    print("  +-------------+------------------+--------------------------+")
    print("  |             |  HEADS             |  LAYERS                  |")
    print("  +-------------+------------------+--------------------------+")
    print("  |  What it    |  Multiple          |  Processing              |")
    print("  |  controls   |  perspectives      |  stages                  |")
    print("  |  How they   |  Work            |  Work                    |")
    print("  |             |  SIMULTANEOUSLY    |  SEQUENTIALLY            |")
    print("  |  Analogy    |  8 experts       |  6 stages of             |")
    print("  |             |  in one room     |  cooking                 |")
    print("  |  Math       |  d_head =          |  Each layer takes        |")
    print("  |             |  d_model / n_heads |  previous layer's output |")
    print("  |  More heads |  Better at       |  Deeper                  |")
    print("  |             |  capturing       |  understanding,          |")
    print("  |             |  diverse         |  more abstract           |")
    print("  |             |  relationships     |  representations         |")
    print("  |  Trade-off  |  Each head has   |  More layers =           |")
    print("  |             |  less dim.       |  More computation        |")
    print("  +-------------+------------------+--------------------------+")
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

    # Explain layers vs heads (important clarification!)
    explain_layers_vs_heads()

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