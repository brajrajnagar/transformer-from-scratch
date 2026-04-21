"""
Layers vs Heads: Understanding the Two Dimensions of Transformers
=================================================================

This is a CLARIFICATION lesson. The Transformer architecture has TWO ways
it scales: DEPTH (layers) and WIDTH (heads). Let's understand the difference.

KEY QUESTION:
- What does "6 encoder layers" mean?
- What does "8 attention heads" mean?
- How do they relate to each other?

ANSWER:
- LAYERS = How MANY times the model processes the input (depth/stages)
- HEADS = How many DIFFERENT perspectives the model uses per layer (width)
"""

import torch
import torch.nn as nn
import math


def explain_layers_vs_heads():
    """Explain the difference between layers and heads clearly."""

    print("=" * 70)
    print("     LAYERS vs HEADS: What's the Difference?")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. ATTENTION HEADS - Multiple Perspectives
    # ------------------------------------------------------------------
    print("1. ATTENTION HEADS = Multiple Perspectives (Width)")
    print("-" * 70)
    print()
    print("Imagine reading a sentence with different EXPERTS, each focused")
    print("on a different aspect of the language:")
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

    # Visual: heads in ONE layer
    print("VISUAL: 8 heads in ONE encoder layer")
    print()
    print("  Input: 'The cat sat on the mat'")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │          SINGLE ENCODER LAYER                   │")
    print("  │                                                 │")
    print("  │  Word:  The   cat   sat   on   the   mat       │")
    print("  │           ↓     ↓     ↓     ↓    ↓     ↓       │")
    print("  │  ┌─────────────────────────────────────────┐   │")
    print("  │  │  Head 1: Grammar relationships          │   │")
    print("  │  │  Head 2: Pronoun resolution             │   │")
    print("  │  │  Head 3: Spatial relationships          │   │")
    print("  │  │  Head 4: Sentiment/emotion              │   │")
    print("  │  │  Head 5: Temporal info                  │   │")
    print("  │  │  Head 6: Entity tracking                │   │")
    print("  │  │  Head 7: Prepositional phrases          │   │")
    print("  │  │  Head 8: Global context                 │   │")
    print("  │  └─────────────────────────────────────────┘   │")
    print("  │           ↓ (concatenate all 8)                 │")
    print("  │  Output: Rich representation of each word      │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    # ------------------------------------------------------------------
    # 2. ENCODER LAYERS - Processing Stages
    # ------------------------------------------------------------------
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

    # Visual: layers as stages
    print("VISUAL: 6 encoder layers processing sequentially")
    print()
    print("  Input: 'The cat sat on the mat'")
    print()
    print("  ┌──────────────────┐")
    print("  │  Layer 1         │  ← Basic word relationships")
    print("  │  (shallow)       │     'cat' is near 'sat' and 'the'")
    print("  └────────┬─────────┘")
    print("           │")
    print("           ▼")
    print("  ┌──────────────────┐")
    print("  │  Layer 2         │  ← Phrase structure")
    print("  │                  │     'the cat' is a noun phrase")
    print("  └────────┬─────────┘")
    print("           │")
    print("           ▼")
    print("  ┌──────────────────┐")
    print("  │  Layer 3         │  ← Sentence meaning")
    print("  │                  │     'cat sat on mat' describes an event")
    print("  └────────┬─────────┘")
    print("           │")
    print("           ▼")
    print("  ┌──────────────────┐")
    print("  │  Layer 4         │  ← Context")
    print("  │                  │     The cat is the subject, mat is the location")
    print("  └────────┬─────────┘")
    print("           │")
    print("           ▼")
    print("  ┌──────────────────┐")
    print("  │  Layer 5         │  ← Deeper context")
    print("  │                  │     This is a simple factual statement")
    print("  └────────┬─────────┘")
    print("           │")
    print("           ▼")
    print("  ┌──────────────────┐")
    print("  │  Layer 6         │  ← Deep understanding")
    print("  │  (deepest)       │     Full contextualized representation")
    print("  └────────┬─────────┘")
    print("           │")
    print("           ▼")
    print("  Output: Rich contextualized embeddings")
    print()

    # ------------------------------------------------------------------
    # 3. THE COMBINATION
    # ------------------------------------------------------------------
    print()
    print("3. COMBINED: Layers × Heads = Total Model Capacity")
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
    print("  - Total attention computations: 6 layers × 8 heads = 48")
    print()

    # Visual: full picture
    print("FULL PICTURE: 6 layers, each with 8 heads")
    print()
    print("  Input")
    print("    │")
    print("    ▼")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  ENCODER LAYER 1                                            │")
    print("  │  ┌─────────────────────────────────────────────────────┐    │")
    print("  │  │ 8 heads: Grammar, Pronouns, Spatial, Sentiment...   │    │")
    print("  │  └─────────────────────────────────────────────────────┘    │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print("    │")
    print("    ▼")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  ENCODER LAYER 2                                            │")
    print("  │  ┌─────────────────────────────────────────────────────┐    │")
    print("  │  │ 8 heads: New perspectives learned from layer 1      │    │")
    print("  │  └─────────────────────────────────────────────────────┘    │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print("    │")
    print("    ▼")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  ... (layers 3, 4, 5) ...                                   │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print("    │")
    print("    ▼")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  ENCODER LAYER 6                                            │")
    print("  │  ┌─────────────────────────────────────────────────────┐    │")
    print("  │  │ 8 heads: Deep, abstract representations              │    │")
    print("  │  └─────────────────────────────────────────────────────┘    │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print("    │")
    print("    ▼")
    print("  Output (deep contextualized)")
    print()

    # ------------------------------------------------------------------
    # 4. AN ANalogy
    # ------------------------------------------------------------------
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
    print("    - Stage 2: Cook (sear, sauté)")
    print("    - Stage 3: Combine (mix ingredients)")
    print("    - Stage 4: Season (add spices)")
    print("    - Stage 5: Plate (beautiful presentation)")
    print("    - Stage 6: Final check (taste test)")
    print()
    print("  Total work: 6 stages × 8 chefs = 48 chef-stages of work!")
    print()

    # ------------------------------------------------------------------
    # 5. NUMERICAL EXAMPLE
    # ------------------------------------------------------------------
    print()
    print("5. WHAT HAPPENS TO ONE WORD: 'cat'")
    print("-" * 70)
    print()

    # Simulate a tiny example
    d_model = 8
    n_heads = 4
    d_head = d_model // n_heads  # 2

    print(f"  d_model = {d_model} (embedding size)")
    print(f"  n_heads = {n_heads} (attention heads)")
    print(f"  d_head = d_model / n_heads = {d_head} (size per head)")
    print()
    print("  Step 1: 'cat' starts as a vector of {d_model} numbers")
    print(f"    embedding = [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]")
    print()
    print(f"  Step 2: Each head sees a DIFFERENT {d_head}-dimensional slice:")
    print()
    print(f"    Head 1 sees: [x₁, x₂]  → focuses on: grammatical role")
    print(f"    Head 2 sees: [x₃, x₄]  → focuses on: entity type")
    print(f"    Head 3 sees: [x₅, x₆]  → focuses on: spatial relation")
    print(f"    Head 4 sees: [x₇, x₈]  → focuses on: sentiment")
    print()
    print("  Step 3: All 8 outputs are CONCATENATED back to {d_model} dims")
    print(f"    combined = [head1_out, head2_out, head3_out, head4_out]")
    print(f"             = [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]")
    print()
    print("  Step 4: This goes through FFN → output of this layer")
    print()
    print("  Step 5: The SAME process repeats in layer 2, 3, 4, 5, 6")
    print("    But each layer LEARNS different weights, so the meaning")
    print("    evolves with each layer.")
    print()

    # ------------------------------------------------------------------
    # 6. SUMMARY TABLE
    # ------------------------------------------------------------------
    print()
    print("6. SUMMARY")
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


def explain_attention_types():
    """Explain the three types of attention in the Transformer."""

    print()
    print("=" * 70)
    print("     THE THREE TYPES OF ATTENTION")
    print("=" * 70)
    print()

    print("1. SELF-ATTENTION (Encoder)")
    print("-" * 40)
    print("   Every token looks at EVERY other token.")
    print()
    print("   Example: 'The cat sat on the mat'")
    print()
    print("          The    cat    sat    on    the    mat")
    print("       ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐")
    print("       │     │ │     │ │     │ │     │ │     │ │     │")
    print("       └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘")
    print("          │      │      │      │      │      │")
    print("          └──────┴──────┴──────┴──────┴──────┘")
    print("                All connect to all")
    print()
    print("   Purpose: Understand context. 'Cat' needs to know about 'the'")
    print("   to understand it's a singular noun.")
    print()

    print("2. MASKED SELF-ATTENTION (Decoder)")
    print("-" * 40)
    print("   Each token can only look at PREVIOUS tokens (no cheating!)")
    print()
    print("   Example: Generating 'Le chat est petit'")
    print()
    print("   Position 0: '<BOS>'     → can see: <BOS>")
    print("   Position 1: 'Le'        → can see: <BOS>, Le")
    print("   Position 2: 'chat'      → can see: <BOS>, Le, chat")
    print("   Position 3: 'est'       → can see: <BOS>, Le, chat, est")
    print("   Position 4: 'petit'     → can see: <BOS>, Le, chat, est, petit")
    print()
    print("   Purpose: Autoregressive generation. We can't look ahead!")
    print()

    print("3. CROSS-ATTENTION (Decoder)")
    print("-" * 40)
    print("   Decoder queries (Q) look at Encoder keys/values (K, V)")
    print()
    print("   Example: Translating 'The cat' → 'Le chat'")
    print()
    print("   Encoder output: [context for 'The', context for 'cat']")
    print()
    print("   When decoder generates 'Le':")
    print("      Q (from decoder) ──► K, V (from encoder 'The')")
    print()
    print("   When decoder generates 'chat':")
    print("      Q (from decoder) ──► K, V (from encoder 'cat')")
    print()
    print("   Purpose: Connect input to output. 'What part of the input")
    print("           do I need to focus on to generate this output word?")
    print()


def main():
    """Main function."""

    print()
    print("+" + "=" * 68 + "+")
    print("|" + " " * 10 + "LAYERS vs HEADS: Clarification Lesson" + " " * 18 + "|")
    print("|" + " " * 20 + "Understanding Transformer Dimensions" + " " * 16 + "|")
    print("+" + "=" * 68 + "+")
    print()

    explain_layers_vs_heads()
    explain_attention_types()

    print("=" * 70)
    print("WHAT'S NEXT:")
    print("-" * 70)
    print("Now that you understand the concepts, Lesson 4 will implement")
    print("the encoder layer with multi-head attention.")
    print()
    print("Run: python 04_encoder_layer.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()