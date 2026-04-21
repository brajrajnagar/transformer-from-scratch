"""
Transformer from Scratch - Lesson 7: Train on Tiny Translation Task
====================================================================

Now we train the full Transformer on a real (tiny) translation task!

WHAT WE'VE BUILT:
  1. Token + Position Embeddings
  2. Multi-Head Self-Attention
  3. Encoder Layer
  4. Decoder Layer
  5. Full Transformer Model

TODAY: Train the model to translate simple English → French sentences.

TASK:
  English: "I am home"
  French:  "Je suis chez moi"

We'll use a tiny vocabulary and a small dataset to keep it simple and fast.

WHAT YOU'LL LEARN:
  - How to prepare data (tokenize, pad, batch)
  - How to train (forward pass, loss, backward pass, optimizer step)
  - How to evaluate (perplexity, accuracy)
  - How to generate (autoregressive decoding)
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# SIMPLE VOCABULARY AND TOKENIZER
# ============================================================================

class SimpleVocab:
    """
    A simple vocabulary for tokenization.

    Maps strings ↔ integers for use with nn.Embedding.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.stoi = {}  # string → int
        self.itos = {}  # int → string
        self.size = 0

        # Reserve special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        # Add special tokens first
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            self.stoi[token] = self.size
            self.itos[self.size] = token
            self.size += 1

    def add_word(self, word: str):
        """Add a word to the vocabulary."""
        if word not in self.stoi and self.size < self.max_size:
            self.stoi[word] = self.size
            self.itos[self.size] = word
            self.size += 1

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        tokens = text.lower().split()
        ids = [self.stoi.get(t, self.stoi[self.unk_token]) for t in tokens]
        # Add BOS and EOS
        return [self.stoi[self.bos_token]] + ids + [self.stoi[self.eos_token]]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.itos.get(i, "<UNK>") for i in ids]
        # Remove BOS and EOS
        tokens = [t for t in tokens if t not in [self.bos_token, self.eos_token]]
        return " ".join(tokens)

    def __len__(self):
        return self.size


# ============================================================================
# TINY DATASET
# ============================================================================

# Tiny English → French parallel corpus
TRAIN_DATA = [
    # Simple greetings
    ("hello", "bonjour"),
    ("hi", "salut"),
    ("good morning", "bonjour"),

    # Identity statements
    ("i am home", "je suis chez moi"),
    ("i am happy", "je suis heureux"),
    ("i am sad", "je suis triste"),
    ("i am tired", "je suis fatigue"),
    ("i am hungry", "j'ai faim"),
    ("i am cold", "j'ai froid"),
    ("i am warm", "j'ai chaud"),

    # Descriptions
    ("the cat is small", "le chat est petit"),
    ("the dog is big", "le chien est grand"),
    ("the cat is big", "le chat est grand"),
    ("the dog is small", "le chien est petit"),
    ("the cat is happy", "le chat est heureux"),
    ("the dog is sad", "le chien est triste"),

    # Locations
    ("i am here", "je suis ici"),
    ("i am there", "je suis la"),
    ("you are here", "tu es ici"),
    ("you are there", "tu es la"),

    # States
    ("it is good", "c'est bon"),
    ("it is bad", "c'est mauvais"),
    ("it is hot", "il fait chaud"),
    ("it is cold", "il fait froid"),

    # More complex
    ("the cat is here", "le chat est ici"),
    ("the dog is there", "le chien est la"),
    ("i am not home", "je ne suis pas chez moi"),
    ("the cat is not here", "le chat n'est pas ici"),
]

# Validation set (slightly different patterns)
VAL_DATA = [
    ("hello", "bonjour"),
    ("i am happy", "je suis heureux"),
    ("the cat is small", "le chat est petit"),
    ("the dog is big", "le chien est grand"),
    ("i am tired", "je suis fatigue"),
    ("it is good", "c'est bon"),
    ("i am here", "je suis ici"),
    ("the cat is happy", "le chat est heureux"),
]


def build_vocab(data_pairs: list[tuple[str, str]], max_size: int = 500) -> tuple[SimpleVocab, SimpleVocab]:
    """Build source and target vocabularies from data."""
    src_vocab = SimpleVocab(max_size)
    tgt_vocab = SimpleVocab(max_size)

    for src_text, tgt_text in data_pairs:
        for word in src_text.lower().split():
            src_vocab.add_word(word)
        for word in tgt_text.lower().split():
            tgt_vocab.add_word(word)

    return src_vocab, tgt_vocab


# ============================================================================
# DATASET
# ============================================================================

class TranslationDataset(Dataset):
    """Simple translation dataset."""

    def __init__(self, data_pairs, src_vocab, tgt_vocab, max_src_len: int = 10, max_tgt_len: int = 10):
        self.data_pairs = data_pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data_pairs[idx]

        # Encode
        src_ids = self.src_vocab.encode(src_text)
        tgt_ids = self.tgt_vocab.encode(tgt_text)

        # Pad/truncate
        src_ids = src_ids[: self.max_src_len]
        tgt_ids = tgt_ids[: self.max_tgt_len]

        # Pad to max length
        src_pad = [self.src_vocab.stoi[self.src_vocab.pad_token]] * (
            self.max_src_len - len(src_ids)
        )
        tgt_pad = [self.tgt_vocab.stoi[self.tgt_vocab.pad_token]] * (
            self.max_tgt_len - len(tgt_ids)
        )

        src_ids += src_pad
        tgt_ids += tgt_pad

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


# ============================================================================
# TRANSFER MODELS (imported from lesson 6)
# ============================================================================

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
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
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q, K, V = self._split_heads(self.W_Q(Q)), self._split_heads(self.W_K(K)), self._split_heads(self.W_V(V))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.attention_dropout(torch.softmax(scores, dim=-1))
        output = self.output_dropout(self.W_O(self._merge_heads(torch.matmul(attn, V))))
        return output, attn


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
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
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, decoder_input, encoder_output, mask=None):
        Q, K, V = self._split_heads(self.W_Q(decoder_input)), self._split_heads(self.W_K(encoder_output)), self._split_heads(self.W_V(encoder_output))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.attention_dropout(torch.softmax(scores, dim=-1))
        output = self.output_dropout(self.W_O(self._merge_heads(torch.matmul(attn, V))))
        return output, attn


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, sizes: tuple[int, ...], eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(sizes))
        self.beta = nn.Parameter(torch.zeros(sizes))
        self.eps = eps

    def forward(self, x):
        mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True, unbiased=False)
        return ((x - mean) / (std + self.eps)) * self.gamma + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))
        self.norm3 = LayerNorm((d_model,))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, causal_mask, padding_mask=None):
        attn_out, _ = self.masked_attention(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        cross_out, _ = self.cross_attention(x, encoder_output, padding_mask)
        x = self.norm2(x + self.dropout(cross_out))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, n_heads=4,
                 n_encoder_layers=2, n_decoder_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = 0
        self.tgt_pad_idx = 0
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)])
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def create_masks(self, src, tgt):
        batch_size, src_seq_len = src.shape
        tgt_seq_len = tgt.shape[1]
        src_padding_mask = (src == self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).to(src.device)
        tgt_padding_mask = (tgt == self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_padding_mask, tgt_causal_mask, tgt_padding_mask

    def encode(self, src):
        x = self.src_pos_encoding(self.src_embedding(src))
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, tgt, encoder_output, src_padding_mask, tgt_causal_mask, tgt_padding_mask):
        x = self.tgt_pos_encoding(self.tgt_embedding(tgt))
        for layer in self.decoder:
            x = layer(x, encoder_output, tgt_causal_mask, src_padding_mask)
        return x

    def forward(self, src, tgt):
        src_pad, tgt_causal, tgt_pad = self.create_masks(src, tgt)
        encoder_output = self.encode(src)
        decoder_output = self.decode(tgt, encoder_output, src_pad, tgt_causal, tgt_pad)
        return self.output_linear(decoder_output)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    for src_batch, tgt_batch in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        optimizer.zero_grad()

        # Forward pass: use tgt_batch[:, :-1] as input, tgt_batch[:, 1:] as target
        # This is teacher forcing: we feed the correct previous tokens
        src_input = src_batch
        tgt_input = tgt_batch[:, :-1]   # Remove <EOS>
        tgt_target = tgt_batch[:, 1:]   # Remove <BOS>

        # Forward pass
        logits = model(src_input, tgt_input)

        # Compute loss
        logits_flat = logits.reshape(-1, logits.shape[-1])
        target_flat = tgt_target.reshape(-1)

        loss = criterion(logits_flat, target_flat)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        preds = logits_flat.argmax(dim=-1)
        correct_tokens += (preds == target_flat).sum().item()
        total_tokens += target_flat.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            src_input = src_batch
            tgt_input = tgt_batch[:, :-1]
            tgt_target = tgt_batch[:, 1:]

            logits = model(src_input, tgt_input)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            target_flat = tgt_target.reshape(-1)

            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item()

            preds = logits_flat.argmax(dim=-1)
            correct_tokens += (preds == target_flat).sum().item()
            total_tokens += target_flat.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens

    return avg_loss, accuracy


def generate_translation(model, src_text, src_vocab, tgt_vocab, max_len: int = 10, device="cpu"):
    """Generate a translation using autoregressive decoding."""
    model.eval()

    # Encode source
    src_ids = src_vocab.encode(src_text)
    src_ids = src_ids[:10]  # Max length
    src_pad_count = 10 - len(src_ids)
    src_ids += [0] * src_pad_count  # Pad with PAD token
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    # Start with <BOS>
    tgt_ids = [tgt_vocab.stoi[tgt_vocab.bos_token]]
    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

    generated = []

    with torch.no_grad():
        # Get encoder output once
        encoder_output = model.encode(src_tensor)

        # Generate one token at a time
        for _ in range(max_len - 1):
            tgt_seq_len = len(tgt_ids)

            # Create causal mask
            causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).to(device)

            # Create padding mask
            src_padding_mask = (src_tensor == 0).unsqueeze(1).unsqueeze(2).to(device)

            # Forward pass
            decoder_output = model.decode(tgt_tensor, encoder_output, src_padding_mask, causal_mask, None)

            # Get last token's logits
            logits = model.output_linear(decoder_output[:, -1, :])
            probs = torch.softmax(logits, dim=-1)

            # Pick the most likely token (greedy decoding)
            next_token = probs.argmax(dim=-1).item()

            # Stop at <EOS>
            if next_token == tgt_vocab.stoi[tgt_vocab.eos_token]:
                break

            tgt_ids.append(next_token)
            generated.append(tgt_vocab.itos.get(next_token, "<UNK>"))

            # Add to tensor for next step
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

    return " ".join(generated)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "TRANSFORMER FROM SCRATCH - LESSON 7" + " " * 28 + "║")
    print("║" + " " * 15 + "Training on Tiny Translation Task" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Build vocabulary
    print("Building vocabulary...")
    src_vocab, tgt_vocab = build_vocab(TRAIN_DATA + VAL_DATA, max_size=200)
    print(f"  Source vocab size: {len(src_vocab)}")
    print(f"  Target vocab size: {len(tgt_vocab)}")
    print()

    # Show vocabulary
    print("Source vocabulary:")
    print(f"  {src_vocab.itos}")
    print()
    print("Target vocabulary:")
    print(f"  {tgt_vocab.itos}")
    print()

    # Create datasets
    print("Creating datasets...")
    train_dataset = TranslationDataset(TRAIN_DATA, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(VAL_DATA, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(VAL_DATA), shuffle=False)

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print()

    # Create model
    print("Creating model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD tokens
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training configuration
    num_epochs = 200
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler
        scheduler.step()

        # Print every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            perplexity = math.exp(min(train_loss, 100))
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Perplexity: {perplexity:.2f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_transformer.pth")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print()

    # Load best model
    model.load_state_dict(torch.load("best_transformer.pth"))

    # Test translations
    print("TESTING TRANSLATIONS")
    print("-" * 70)
    print()

    test_sentences = [
        "i am home",
        "the cat is small",
        "i am happy",
        "it is cold",
        "the dog is big",
        "i am here",
        "hello",
    ]

    for eng in test_sentences:
        french = generate_translation(model, eng, src_vocab, tgt_vocab, device=device)
        # Find the expected translation
        expected = None
        for src, tgt in TRAIN_DATA + VAL_DATA:
            if src == eng:
                expected = tgt
                break

        print(f"  English:    '{eng}'")
        print(f"  Predicted:  '{french}'")
        if expected:
            print(f"  Expected:   '{expected}'")
        print()

    # Show attention patterns (if possible)
    print("=" * 70)
    print("WHAT WE LEARNED:")
    print("-" * 70)
    print("✓ Tokenizer: Simple word-based tokenization")
    print("✓ Dataset: PyTorch Dataset for translation pairs")
    print("✓ Training: Forward pass → Loss → Backward pass → Optimizer step")
    print("✓ Teacher Forcing: Feed correct previous tokens during training")
    print("✓ Greedy Decoding: Pick highest probability token during inference")
    print("✓ Early Stopping: Save best model based on validation loss")
    print()
    print("KEY CONCEPTS:")
    print("  1. Teacher Forcing: During training, feed the ACTUAL target tokens")
    print("     as input to the decoder (not the model's predictions). This")
    print("     makes training much faster and more stable.")
    print()
    print("  2. Greedy Decoding: During inference, pick the token with the")
    print("     highest probability at each step. Simple but effective.")
    print()
    print("  3. Greedy decoding is autoregressive: each token depends on the")
    print("     previously generated tokens.")
    print()


if __name__ == "__main__":
    main()