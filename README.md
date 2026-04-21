# Transformer from Scratch 🧠

Build and train a Transformer model from the ground up, learning every component along the way.

## 📚 Learning Path

Run each lesson in order. Each file is a standalone script with detailed explanations.

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

## 🏗️ Architecture Diagram

The Transformer follows an **encoder-decoder** architecture (Vaswani et al., 2017):

```
INPUT: "The cat sat"                    TARGET: "<BOS> Le chat ... <EOS>"
         │                                      │
         ▼                                      ▼
   ┌──────────────┐                        ┌──────────────┐
   │  EMBEDDING   │                        │  EMBEDDING   │
   │ + POSITION   │                        │ + POSITION   │
   └──────┬───────┘                        └──────┬───────┘
          │                                       │
          ▼                                       ▼
   ┌──────────────┐                        ┌──────────────┐
   │   ENCODER    │                        │   DECODER    │
   │  (N layers)  │                        │  (N layers)  │
   │              │                        │              │
   │ ┌──────────┐ │                        │ ┌──────────┐ │
   │ │Self-Attn │ │                        │ │Masked    │ │
   │ │          │ │                        │ │Self-Attn │ │
   │ └──────────┘ │                        │ └──────────┘ │
   │      │       │                        │      │       │
   │ ┌──────────┐ │                        │ ┌──────────┐ │
   │ │  FFN     │ │                        │ │Cross-     │ │
   │ └──────────┘ │                        │ │Attention  │ │
   │              │                        │ │          │ │
   │              │                        │ └────┬─────┘ │
   └──────┬───────┘                        └────┬─┴──────┘
          │                                     │
          │  Encoder Output                     │  Decoder Output
          │  (contextualized                    │  (token logits
          │   representations)                  │  over vocab)
          ▼                                     ▼
   ┌──────────────┐                        ┌──────────────┐
   │              │                        │  Linear +    │
   │              │                        │  Softmax     │
   └──────────────┘                        └──────────────┘

   ═══════════════════════════════════════════════════════
   Cross-Attention: Decoder queries (Q) attend to
   Encoder keys/values (K, V) — flows ↓ from encoder
   ═══════════════════════════════════════════════════════
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