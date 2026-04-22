# Transformer from Scratch 🧠

Build and train a Transformer model from the ground up, learning every component along the way.

## 📚 Learning Path

Run each lesson in order. Each file is a standalone script with detailed explanations and comprehensive docstrings with examples.

```bash
# Activate the virtual environment first:
source .venv/bin/activate

# Then run each lesson:
python 01_architecture_overview.py
python 02_embeddings.py
python 03_multi_head_attention.py
python 04_encoder_layer.py
python 05_decoder_layer.py
python 06_full_transformer.py
python 07_train_translate.py
python 08_iwslt_vi_en.py --demo      # small built-in EN↔VI pairs
python 08_iwslt_vi_en.py --full      # full IWSLT corpus (~133K pairs)
python 08_iwslt_vi_en.py --infer     # interactive prompt using the saved model
```

## 🗂️ Lessons

| # | File | Topic | Key Concepts |
|---|------|-------|--------------|
| 1 | `01_architecture_overview.py` | Architecture Overview | Encoder-Decoder pipeline, attention diagram, model sizing |
| 2 | `02_embeddings.py` | Embeddings | Token embeddings, positional encoding (sinusoidal) |
| 3 | `03_multi_head_attention.py` | Multi-Head Attention | Scaled dot-product attention, head splitting/merging |
| 4 | `04_encoder_layer.py` | Encoder Layer | Self-attention, FFN, residual connections, layer norm |
| 5 | `05_decoder_layer.py` | Decoder Layer | Masked attention, cross-attention, 3 sublayers |
| 6 | `06_full_transformer.py` | Full Model | Putting everything together, masks, forward pass |
| 7 | `07_train_translate.py` | Training | Data prep, teacher forcing, greedy decoding, translation |
| 8 | `08_iwslt_vi_en.py` | Real Dataset | IWSLT EN↔VI corpus, BPE tokenizer, LR warmup + cosine decay, gradient clipping, interactive inference |

## 🏗️ Architecture Diagram

The Transformer follows an **encoder-decoder** architecture (Vaswani et al., 2017):

```
                    ENCODER                          DECODER
   ──────────────────────────              ──────────────────────────
   Input: "The cat sat"                    Output: "Le chat est assis"
                                               │
                                               ▼
   ┌─────────────┐                      ┌─────────────┐
   │   INPUT     │                      │   OUTPUT    │
   │  EMBEDDING  │                      │  EMBEDDING  │
   │  + POS ENC  │                      │  + POS ENC  │
   └──────┬──────┘                      └──────┬──────┘
          │                                    │
          ▼                                    ▼
   ┌─────────────────┐                  ┌─────────────────┐
   │   ENCODER       │                  │   DECODER       │
   │   LAYER 1       │                  │   LAYER 1       │
   │ ┌─────────────┐ │                  │ ┌─────────────┐ │
   │ │ Multi-Head  │ │                  │ │ Masked      │ │
   │ │ Self-Attn   │ │                  │ │ Self-Attn   │ │
   │ │ Q=K=V=input │ │                  │ │ Q=input     │ │
   │ └──────┬──────┘ │                  │ │ (causal)    │ │
   │        │        │                  │ └──────┬──────┘ │
   │   ┌────▼────┐   │                  │        │        │
   │   │Add+Norm │   │                  │   ┌────▼────┐   │
   │   └────┬────┘   │                  │   │Add+Norm │   │
   │        │        │                  │   └────┬────┘   │
   │   ┌────▼────┐   │                  │        │        │
   │   │   FFN   │   │                  │   ┌────▼────┐   │
   │   └────┬────┘   │                  │   │ Cross-   │  │
   │        │        │                  │   │ Attention│  │
   │   ┌────▼────┐   │       __________ │   │ K,V=enc  │  │
   │   │Add+Norm │   │      |           │   │ Q=dec   │   │
   │   └────┬────┘   │      |           │   └────┬────┘   │
   └────────┼────────┘      |           │        │        │
            │               |           │   ┌────▼────┐   │
            │               |           │   │Add+Norm │   │
            │               |           │   └────┬────┘   │
            ▼               |           │        │        │
   ┌─────────────────┐      |           │   ┌────▼────┐   │
   │   ENCODER       │      |           │   │   FFN   │   │
   │   LAYER N       │      |           │   └────┬────┘   │
   │   (same as L1)  │      |           │        │        │
   └────────┬────────┘      |           │   ┌────▼────┐   │
            │               |           │   │Add+Norm │   │
            │               |           │   └────┬────┘   │
            │               |           └────────┼────────┘
            │               |                    │
            │          Encoder Output            │
            │      (context vectors K,V)         │
            │               │                    │
            │               │                    ▼
            │               │          ┌─────────────────┐
            │               │          │   LINEAR +      │
            │               │          │   SOFTMAX       │
            │               │          └────────┬────────┘
            │               │                   │
            ▼               │                   ▼
   Final Encoder            │          Output Probabilities
   Representations ---------             over vocabulary
                                               │
                                               ▼
                                      Select most likely
                                      token (argmax)
                                               │
                                               ▼
                                      Generated Translation
                                      (fed back as input
                                       for next decoding step)
                            
   Note: In cross-attention:
   - Q (Query) comes from DECODER (current decoder output)
   - K (Keys) and V (Values) come from ENCODER output
```

Each encoder/decoder block contains **N=6 identical layers** (in the base model).
Inside each layer:
- **Encoder**: Self-Attention → Add+Norm → FFN → Add+Norm
- **Decoder**: Masked Self-Attention → Add+Norm → Cross-Attention → Add+Norm → FFN → Add+Norm

## 🔧 Setup

```bash
# Create virtual environment (already done)
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install torch numpy

# Lesson 8 also needs:
pip install tokenizers tqdm
```

## 📖 Key Concepts Explained

### Attention
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```
- **Q (Query)**: What am I looking for?
- **K (Keys)**: What information do I have?
- **V (Values)**: The actual content to retrieve

### Causal Masking (Decoder)
```
Position 0: can see [0]
Position 1: can see [0, 1]
Position 2: can see [0, 1, 2]
...
```
Prevents the decoder from "looking ahead" at future tokens during training.

### Encoder vs Decoder
| | Encoder | Decoder |
|---|---------|---------|
| Self-Attention | Full (all→all) | Masked (backward only) |
| Cross-Attention | N/A | Yes (queries encoder) |
| Purpose | Understand input | Generate output |

### How Masking Works in MultiHeadAttention

The masking mechanism is implemented in `MultiHeadAttention.forward()` and applies to the `seq_len × seq_len` attention scores matrix:

```python
# scores shape: (batch, n_heads, seq_len, seq_len)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

# Apply mask: where mask=0, set score to -1e9 (negative infinity)
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

# After softmax, attention to masked positions becomes ~0
attn = torch.softmax(scores, dim=-1)
```

**Mask Types:**

1. **src_padding_mask** (Encoder): Masks padding positions in source
   - Shape: `(batch, 1, 1, src_len)`
   - Columns for pad positions get -1e9 → attention weights become 0.0

2. **tgt_causal_mask** (Decoder self-attention): Lower triangular matrix
   - Shape: `(tgt_len, tgt_len)`
   - Upper triangle (future positions) masked with -1e9

3. **Combined mask** (Decoder): `causal_mask AND padding_mask`
   - Masks both future positions AND padding positions

4. **src_padding_mask in Cross-Attention**: Decoder attending to encoder
   - Same mechanism: encoder pad positions masked

## 🎯 Results

Lesson 7 trains the Transformer to translate 28 tiny English→French sentences. After 200 epochs:

- **Training loss**: 0.0075 (near zero)
- **Validation loss**: 0.0012 (near zero)
- **Translation accuracy**: 100% (7/7 correct)

```
English:    'i am home'      → French:  'je suis chez moi'    ✓
English:    'the cat is small' → French:  'le chat est petit'   ✓
English:    'i am happy'     → French:  'je suis heureux'     ✓
English:    'it is cold'     → French:  'il fait froid'       ✓
English:    'the dog is big' → French:  'le chien est grand'  ✓
English:    'i am here'      → French:  'je suis ici'         ✓
English:    'hello'          → French:  'bonjour'             ✓
```

### Lesson 8 — Real Dataset (IWSLT Vietnamese–English)

Lesson 8 trains on the full IWSLT EN↔VI corpus (~133K sentence pairs) with a real subword tokenizer (BPE via HuggingFace `tokenizers`). Data is sourced from [stefan-it/nmt-en-vi](https://github.com/stefan-it/nmt-en-vi) — drop the `.tgz` into `data/` and the script extracts it automatically on first run. It adds:

- **LR warmup + cosine decay** (fixed so the schedule doesn't compound onto its own output — a bug that silently pinned LR at 0)
- **Padding masks** applied to encoder self-attention and decoder self/cross-attention, with a consistent `1 = attend-to` convention
- **Gradient clipping** (max-norm = 1.0)
- **Checkpoint + tokenizer persistence** so you can reload without retraining:
  - `best_iwslt_transformer.pth` — model weights + architecture config
  - `en_tokenizer.json` / `vi_tokenizer.json` — saved BPE tokenizers
- **`--infer` mode** — drops into an interactive EN→VI prompt:

```
$ python 08_iwslt_vi_en.py --infer
EN> i am happy
VI> tôi vui
EN>
```

## 📐 Model Configurations

| Config | d_model | Heads | d_ff | Layers | Params |
|--------|---------|-------|------|--------|--------|
| Tiny | 64 | 4 | 128 | 2+2 | ~174K |
| Small | 256 | 8 | 512 | 4+4 | ~5M |
| Base | 512 | 8 | 2048 | 6+6 | ~125M |
| Big | 1024 | 16 | 4096 | 6+6 | ~220M |

## 📝 Notes

- All code is from scratch using PyTorch (no `torch.nn.Transformer`)
- Each file is self-contained and runnable
- Comments explain every design decision
- The training in Lesson 7 uses a tiny dataset (28 examples) for fast demonstration
- Lesson 8 scales up to a real corpus (IWSLT EN↔VI) with subword tokenization and an interactive inference mode

## 📚 Documentation

All classes and functions include comprehensive docstrings with:
- Detailed explanations of purpose and behavior
- Parameter descriptions with types and shapes
- Return value documentation
- Practical code examples showing real usage

To explore the documentation interactively in Python:

```python
from 08_iwslt_vi_en import MultiHeadAttention, Transformer

# View class documentation
help(MultiHeadAttention)
help(MultiHeadAttention.forward)

# View Transformer documentation
help(Transformer)
help(Transformer.encode)
help(Transformer.decode)
```

Or use your IDE's built-in documentation viewer (e.g., hover over a class/function in VS Code).