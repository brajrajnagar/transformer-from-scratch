"""
Transformer from Scratch - Lesson 8: IWSLT Vietnamese-English Translation
=========================================================================

Training on a REAL dataset: IWSLT (ACL International Workshop on Language
Resources and Language Translation) Vietnamese-English corpus.

DATASET STATS:
  Training: ~133,317 sentence pairs
  Validation: ~1,553 sentence pairs
  Test: ~1,268 sentence pairs

WHAT YOU'LL LEARN:
  1. Subword tokenization with HuggingFace tokenizers (BPE)
  2. Loading real translation data from files
  3. Proper data loading with batching and padding
  4. Learning rate scheduling (warmup + cosine decay)
  5. Gradient clipping for stable training
  6. Greedy decoding

HOW TO RUN:
  # Full dataset with HuggingFace tokenizers:
  python 08_iwslt_vi_en.py --full

  # Demo mode with small subset (works on CPU):
  python 08_iwslt_vi_en.py --demo

  # First install dependencies:
  pip install tokenizers tqdm
"""

import argparse
import math
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from tokenizers.normalizers import Normalizer, Sequence
    from tokenizers.pre_tokenizers import Whitespace
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False
    print("WARNING: 'tokenizers' library not installed.")
    print("Install with: pip install tokenizers")
    print("Falling back to simple BPE implementation...")


CHECKPOINT_PATH = "best_iwslt_transformer.pth"


def tokenizer_paths():
    """Paths for the en/vi tokenizers — .json for HF, .pkl for the simple fallback."""
    ext = "json" if HAS_TOKENIZERS else "pkl"
    return f"en_tokenizer.{ext}", f"vi_tokenizer.{ext}"


# ============================================================================
# HUGGINGFACE BPE TOKENIZER WRAPPER
# ============================================================================

class HuggingFaceBPE:
    """
    Wrapper around HuggingFace's tokenizers.BPE for easy training and encoding.

    WHY USE HUGGINGFACE TOKENIZERS?
    ───────────────────────────────
    1. Well-tested and optimized implementation
    2. Proper handling of special tokens
    3. Supports multiple tokenization algorithms (BPE, WordPiece, Unigram)
    4. Fast encoding/decoding
    5. Can save/load tokenizer configuration

    BPE (Byte Pair Encoding) works by:
    1. Starting with characters as the base vocabulary
    2. Iteratively merging the most frequent adjacent pairs
    3. Creating a compact vocabulary that handles OOV words

    Example:
      "unhappiness" → ["un", "happi", "ness"]
      Even if "unhappiness" was never seen, BPE breaks it into known subwords.
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def train(self, files: list[str]):
        """
        Train BPE tokenizer on text files.

        Args:
            files: List of file paths to train on
        """
        if not HAS_TOKENIZERS:
            raise ImportError("HuggingFace tokenizers library required")

        # Initialize empty tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Configure pre-tokenization (split on whitespace first)
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Create trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # Train on files
        print(f"  Training on {len(files)} file(s)...")
        self.tokenizer.train(files, trainer=trainer)

        print(f"  Final vocabulary size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet")
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet")
        return self.tokenizer.decode(ids)

    @property
    def vocab_len(self) -> int:
        """Get vocabulary length."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_len

    def save(self, path: str):
        """Save tokenizer to file."""
        if self.tokenizer:
            self.tokenizer.save(path)

    def load(self, path: str):
        """Load tokenizer from file."""
        if HAS_TOKENIZERS:
            self.tokenizer = Tokenizer.from_file(path)


# ============================================================================
# SIMPLE BPE TOKENIZER (fallback if tokenizers not installed)
# ============================================================================

class SimpleBPE:
    """Simple BPE implementation as fallback."""

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.symbols = []
        self.symbols_to_id = {}
        self.merges = []
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def _get_symbols(self, text: str) -> list[str]:
        symbols = list(text) + ["<w>"]
        return symbols

    def _get_pairs(self, words: list[list[str]]) -> Counter:
        pairs = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def train(self, texts: list[str]):
        word_freqs = Counter()
        all_words = []

        for text in texts:
            tokens = text.split()
            for token in tokens:
                if token.strip():
                    symbols = self._get_symbols(token)
                    word_freqs[" ".join(symbols)] += 1
                    all_words.append(symbols)

        num_merges = self.vocab_size - 256
        num_merges = min(num_merges, 5000)

        all_chars = set()
        for word in all_words:
            all_chars.update(word)
        self.symbols = sorted(all_chars)
        self.symbols_to_id = {s: i for i, s in enumerate(self.symbols)}

        for merge_idx in range(num_merges):
            pairs = self._get_pairs(all_words)

            best_pair = None
            best_count = 0
            for pair, count in pairs.items():
                if count >= 2 and count > best_count:
                    best_pair = pair
                    best_count = count

            if best_pair is None:
                break

            new_symbol = "".join(best_pair)
            self.symbols.append(new_symbol)
            self.symbols_to_id[new_symbol] = len(self.symbols) - 1
            self.merges.append(best_pair)

            new_all_words = []
            for word in all_words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_symbol)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_all_words.append(new_word)
            all_words = new_all_words

            if (merge_idx + 1) % 500 == 0:
                print(f"  Trained {merge_idx + 1}/{num_merges} merges")

    def encode(self, text: str) -> list[int]:
        tokens = text.split()
        result = []

        for token in tokens:
            if token in self.symbols_to_id:
                result.append(self.symbols_to_id[token])
            else:
                i = 0
                while i < len(token):
                    found = False
                    for j in range(min(len(token) - i, 10), 0, -1):
                        sub = token[i:i + j]
                        if sub in self.symbols_to_id:
                            result.append(self.symbols_to_id[sub])
                            i += j
                            found = True
                            break
                    if not found:
                        result.append(self.symbols_to_id.get("<unk>", 1))
                        i += 1

        return result

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for token_id in ids:
            if token_id in self.symbols:
                token_str = self.symbols[token_id]
                if token_str not in ("<pad>", "<unk>", "<bos>", "<eos>", "<w>"):
                    tokens.append(token_str.replace("<w>", ""))
        return " ".join(tokens)

    @property
    def vocab_len(self) -> int:
        return len(self.symbols)

    def get_vocab_size(self) -> int:
        return self.vocab_len

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "symbols": self.symbols,
                "symbols_to_id": self.symbols_to_id,
                "merges": self.merges,
                "special_tokens": self.special_tokens,
                "vocab_size": self.vocab_size,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.symbols = state["symbols"]
        self.symbols_to_id = state["symbols_to_id"]
        self.merges = state["merges"]
        self.special_tokens = state["special_tokens"]
        self.vocab_size = state["vocab_size"]


# ============================================================================
# DEMO DATASET (small subset for immediate testing)
# ============================================================================

DEMO_TRAIN_DATA = [
    ("hello", "xin chào"),
    ("hi", "chào"),
    ("good morning", "chào buổi sáng"),
    ("good afternoon", "chào buổi chiều"),
    ("good evening", "chào buổi tối"),
    ("how are you", "bạn khỏe không"),
    ("i am fine", "tôi khỏe"),
    ("thank you", "cảm ơn bạn"),
    ("thanks", "cảm ơn"),
    ("you are welcome", "không có gì"),
    ("goodbye", "tạm biệt"),
    ("see you", "hẹn gặp lại"),
    ("i am home", "tôi ở nhà"),
    ("i am happy", "tôi vui"),
    ("i am sad", "tôi buồn"),
    ("i am tired", "tôi mệt"),
    ("i am hungry", "tôi đói"),
    ("i am cold", "tôi lạnh"),
    ("i am warm", "tôi ấm"),
    ("i am busy", "tôi bận"),
    ("i am free", "tôi rảnh"),
    ("i am ready", "tôi sẵn sàng"),
    ("i am here", "tôi ở đây"),
    ("i am there", "tôi ở đó"),
    ("i am learning", "tôi đang học"),
    ("i am working", "tôi đang làm việc"),
    ("i am sleeping", "tôi đang ngủ"),
    ("i am eating", "tôi đang ăn"),
    ("i am drinking", "tôi đang uống"),
    ("i am reading", "tôi đang đọc"),
    ("i am writing", "tôi đang viết"),
    ("the cat is small", "con mèo nhỏ"),
    ("the dog is big", "con chó lớn"),
    ("the cat is big", "con mèo lớn"),
    ("the dog is small", "con chó nhỏ"),
    ("the cat is happy", "con mèo vui"),
    ("the dog is sad", "con chó buồn"),
    ("the book is good", "cuốn sách tốt"),
    ("the food is delicious", "đồ ăn ngon"),
    ("the weather is hot", "thời tiết nóng"),
    ("the weather is cold", "thời tiết lạnh"),
    ("the house is big", "ngôi nhà lớn"),
    ("the house is small", "ngôi nhà nhỏ"),
    ("the car is fast", "xe hơi nhanh"),
    ("the car is slow", "xe hơi chậm"),
    ("the river is long", "con sông dài"),
    ("the mountain is high", "ngọn núi cao"),
    ("it is good", "nó tốt"),
    ("it is bad", "nó xấu"),
    ("it is hot", "trời nóng"),
    ("it is cold", "trời lạnh"),
    ("i love you", "tôi yêu bạn"),
    ("i dont know", "tôi không biết"),
    ("i understand", "tôi hiểu"),
    ("i dont understand", "tôi không hiểu"),
    ("please", "làm ơn"),
    ("sorry", "xin lỗi"),
    ("excuse me", "xin lỗi"),
    ("how much", "bao nhiêu"),
    ("very good", "rất tốt"),
    ("very bad", "rất xấu"),
    ("very hot", "rất nóng"),
    ("very cold", "rất lạnh"),
    ("very big", "rất lớn"),
    ("very small", "rất nhỏ"),
    ("very happy", "rất vui"),
    ("very sad", "rất buồn"),
    ("what is this", "đây là gì"),
    ("what is that", "đó là gì"),
    ("who are you", "bạn là ai"),
    ("where are you from", "bạn đến từ đâu"),
    ("how old are you", "bạn bao tuổi"),
    ("why are you sad", "tại sao bạn buồn"),
    ("what time is it", "mấy giờ rồi"),
    ("can you help me", "bạn có thể giúp tôi không"),
    ("do you speak english", "bạn có nói tiếng anh không"),
    ("is it far", "có xa không"),
    ("is it near", "có gần không"),
    ("the cat is here", "con mèo ở đây"),
    ("the dog is there", "con chó ở đó"),
    ("i want to go", "tôi muốn đi"),
    ("i want to eat", "tôi muốn ăn"),
    ("i want to sleep", "tôi muốn ngủ"),
    ("i want to learn", "tôi muốn học"),
    ("this is my house", "đây là nhà của tôi"),
    ("this is my book", "đây là sách của tôi"),
    ("that is my car", "đó là xe của tôi"),
    ("the sun is shining", "mặt trời đang chiếu"),
    ("the rain is falling", "mưa đang rơi"),
    ("the wind is blowing", "gió đang thổi"),
]

DEMO_VAL_DATA = [
    ("good night", "chúc ngủ ngon"),
    ("i am lost", "tôi bị lạc"),
    ("the food is good", "đồ ăn tốt"),
    ("i am coming", "tôi đang đến"),
    ("i am going", "tôi đang đi"),
    ("how are you today", "hôm nay bạn thế nào"),
    ("nice to meet you", "rất vui được gặp bạn"),
    ("have a nice day", "chúc bạn ngày mới tốt lành"),
    ("the city is beautiful", "thành phố đẹp"),
    ("the people are friendly", "mọi người thân thiện"),
]


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_demo_data():
    """Load the small demo dataset for quick testing."""
    print("  Using demo dataset (~102 training pairs)")
    return DEMO_TRAIN_DATA, DEMO_VAL_DATA


def load_full_data():
    """
    Load the full IWSLT Vietnamese-English dataset from local files.

    Expected files in data/ directory:
      - train.en (English training sentences)
      - train.vi (Vietnamese training sentences)
      - dev.en / dev.vi (validation, optional)
      - tst2012.en / tst2012.vi (test, optional)
    """
    data_dir = Path("data")

    train_en_path = data_dir / "train.en"
    train_vi_path = data_dir / "train.vi"
    dev_en_path = data_dir / "dev.en"
    dev_vi_path = data_dir / "dev.vi"

    # Check if extracted, if not extract from tgz
    if not (train_en_path.exists() and train_vi_path.exists()):
        import tarfile
        print("  Extracting training data...")
        tgz_path = data_dir / "train-en-vi.tgz"
        if tgz_path.exists():
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
            print("  Extracted train-en-vi.tgz")
        else:
            raise FileNotFoundError("Could not find training data")

    # Read training pairs
    print("  Reading training data...")
    with open(train_en_path, "r", encoding="utf-8") as f:
        en_train = [line.strip() for line in f if line.strip()]
    with open(train_vi_path, "r", encoding="utf-8") as f:
        vi_train = [line.strip() for line in f if line.strip()]

    # Read validation pairs
    print("  Reading development data...")
    if dev_en_path.exists() and dev_vi_path.exists():
        with open(dev_en_path, "r", encoding="utf-8") as f:
            en_val = [line.strip() for line in f if line.strip()]
        with open(dev_vi_path, "r", encoding="utf-8") as f:
            vi_val = [line.strip() for line in f if line.strip()]
        val_pairs = list(zip(en_val, vi_val))
    else:
        print("  No dev data found, using 1% of training data as validation")
        split_idx = int(len(en_train) * 0.99)
        en_val = en_train[split_idx:]
        vi_val = vi_train[split_idx:]
        val_pairs = list(zip(en_val, vi_val))
        en_train = en_train[:split_idx]
        vi_train = vi_train[:split_idx]

    train_pairs = list(zip(en_train, vi_train))

    print(f"  Train: {len(train_pairs):,} pairs")
    print(f"  Validation: {len(val_pairs):,} pairs")
    print()

    # Show sample
    print("  Sample training pairs:")
    for i in range(min(3, len(train_pairs))):
        en_preview = train_pairs[i][0][:70]
        vi_preview = train_pairs[i][1][:70]
        print(f"    EN: {en_preview}")
        print(f"    VI: {vi_preview}")
        print()

    return train_pairs, val_pairs


# ============================================================================
# DATASET CLASS
# ============================================================================

class TranslationDataset(Dataset):
    """Dataset for Vietnamese-English translation."""

    def __init__(self, en_texts: list[str], vi_texts: list[str],
                 en_vocab, vi_vocab, max_len: int = 50):
        self.en_vocab = en_vocab
        self.vi_vocab = vi_vocab
        self.max_len = max_len

        self.en_encoded = []
        self.vi_encoded = []

        # Get special token IDs
        self.bos_id = 2  # <bos>
        self.eos_id = 3  # <eos>
        self.pad_id = 0  # <pad>

        for en_text, vi_text in zip(en_texts, vi_texts):
            en_ids = en_vocab.encode(en_text)
            vi_ids = vi_vocab.encode(vi_text)

            # Add BOS and EOS
            en_ids = [self.bos_id] + en_ids[:max_len - 2] + [self.eos_id]
            vi_ids = [self.bos_id] + vi_ids[:max_len - 2] + [self.eos_id]

            # Pad to max_len
            en_ids = en_ids[:max_len] + [self.pad_id] * max(0, max_len - len(en_ids))
            vi_ids = vi_ids[:max_len] + [self.pad_id] * max(0, max_len - len(vi_ids))

            self.en_encoded.append(en_ids)
            self.vi_encoded.append(vi_ids)

    def __len__(self) -> int:
        return len(self.en_encoded)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.en_encoded[idx], dtype=torch.long),
            torch.tensor(self.vi_encoded[idx], dtype=torch.long),
        )


# ============================================================================
# TRANSFORMER MODEL
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

    def forward(self, x, src_padding_mask=None):
        attn_out, _ = self.self_attention(x, x, x, src_padding_mask)
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
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_heads=8,
                 n_encoder_layers=3, n_decoder_layers=3, d_ff=256, dropout=0.1):
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
        # Convention: 1 = attend-to, 0 = mask-out (matches the causal mask).
        src_padding_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).to(src.device)
        tgt_padding_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_padding_mask, tgt_causal_mask, tgt_padding_mask

    def encode(self, src, src_padding_mask=None):
        x = self.src_pos_encoding(self.src_embedding(src))
        for layer in self.encoder:
            x = layer(x, src_padding_mask)
        return x

    def decode(self, tgt, encoder_output, src_padding_mask, tgt_causal_mask, tgt_padding_mask):
        x = self.tgt_pos_encoding(self.tgt_embedding(tgt))
        # Combine causal + target padding so decoder self-attn also ignores pad positions.
        self_attn_mask = tgt_causal_mask
        if tgt_padding_mask is not None:
            self_attn_mask = tgt_causal_mask.bool() & tgt_padding_mask.bool()
        for layer in self.decoder:
            x = layer(x, encoder_output, self_attn_mask, src_padding_mask)
        return x

    def forward(self, src, tgt):
        src_pad, tgt_causal, tgt_pad = self.create_masks(src, tgt)
        encoder_output = self.encode(src, src_pad)
        decoder_output = self.decode(tgt, encoder_output, src_pad, tgt_causal, tgt_pad)
        return self.output_linear(decoder_output)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_lr_scheduler(base_lr: float, num_warmup_steps: int, current_step: int, total_steps: int):
    """Learning rate scheduler with warmup + cosine decay.

    Takes a fixed base_lr so the schedule is applied relative to it, not to the
    already-decayed value in the optimizer (which would compound each step).
    """
    if num_warmup_steps <= 0:
        progress = current_step / max(1, total_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    if current_step < num_warmup_steps:
        # Start from a small nonzero LR so step 0 still moves weights.
        return base_lr * ((current_step + 1) / num_warmup_steps)
    else:
        progress = (current_step - num_warmup_steps) / max(1, total_steps - num_warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, dataloader, optimizer, criterion, device, current_step: int,
                total_steps: int, warmup_steps: int, base_lr: float,
                epoch: int, total_epochs: int):
    """Train for one epoch with learning rate scheduling."""
    model.train()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    # Progress bar for training
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for src_batch, tgt_batch in pbar:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        optimizer.zero_grad()

        src_input = src_batch
        tgt_input = tgt_batch[:, :-1]
        tgt_target = tgt_batch[:, 1:]

        logits = model(src_input, tgt_input)

        logits_flat = logits.reshape(-1, logits.shape[-1])
        target_flat = tgt_target.reshape(-1)
        loss = criterion(logits_flat, target_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        new_lr = get_lr_scheduler(base_lr, warmup_steps, current_step, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        optimizer.step()

        total_loss += loss.item()
        preds = logits_flat.argmax(dim=-1)
        correct_tokens += (preds == target_flat).sum().item()
        total_tokens += target_flat.numel()

        current_step += 1

        # Update progress bar
        running_loss = total_loss / (pbar.n + 1)
        running_acc = correct_tokens / total_tokens
        pbar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "acc": f"{running_acc:.4f}",
            "lr": f"{new_lr:.2e}"
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens

    return avg_loss, accuracy, current_step


def evaluate(model, dataloader, criterion, device, epoch: int = None, total_epochs: int = None):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    # Progress bar for validation
    desc = f"Epoch {epoch}/{total_epochs} [Val]" if epoch else "Validation"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for src_batch, tgt_batch in pbar:
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

            # Update progress bar
            running_loss = total_loss / (pbar.n + 1)
            running_acc = correct_tokens / total_tokens
            pbar.set_postfix({
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_acc:.4f}"
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens

    return avg_loss, accuracy


def generate_translation(model, src_text, en_vocab, vi_vocab, max_len: int = 50, device="cpu"):
    """Generate a translation using greedy decoding."""
    model.eval()

    bos_id = 2
    eos_id = 3
    pad_id = 0

    # Encode source
    encoded = en_vocab.encode(src_text.lower())
    encoded = [bos_id] + encoded[:max_len - 2] + [eos_id]
    encoded = encoded[:max_len]
    encoded += [pad_id] * (max_len - len(encoded))

    src_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    # Start with <BOS>
    tgt_ids = [bos_id]
    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

    generated_tokens = []

    with torch.no_grad():
        src_padding_mask = (src_tensor != pad_id).unsqueeze(1).unsqueeze(2).to(device)
        encoder_output = model.encode(src_tensor, src_padding_mask)

        for _ in range(max_len - 1):
            tgt_seq_len = len(tgt_ids)
            causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).to(device)

            decoder_output = model.decode(tgt_tensor, encoder_output, src_padding_mask, causal_mask, None)

            logits = model.output_linear(decoder_output[:, -1, :])
            probs = torch.softmax(logits, dim=-1)

            next_token = probs.argmax(dim=-1).item()

            if next_token == eos_id:
                break

            tgt_ids.append(next_token)
            generated_tokens.append(next_token)

            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

    # Decode to text
    decoded = vi_vocab.decode(generated_tokens)
    return decoded


# ============================================================================
# INFERENCE
# ============================================================================

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_loop():
    """Load checkpoint + tokenizers and drop into an interactive EN→VI prompt."""
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint at {CHECKPOINT_PATH}. Train first with --demo or --full.")
        return

    en_path, vi_path = tokenizer_paths()
    if not (os.path.exists(en_path) and os.path.exists(vi_path)):
        print(f"Missing tokenizer files ({en_path}, {vi_path}). Train first.")
        return

    device = pick_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    tokenizer_type = checkpoint["tokenizer_type"]
    cfg = checkpoint["model_config"]

    if tokenizer_type == "huggingface":
        if not HAS_TOKENIZERS:
            print("Checkpoint used HuggingFace tokenizers but that library isn't installed.")
            return
        en_vocab = HuggingFaceBPE()
        vi_vocab = HuggingFaceBPE()
    else:
        en_vocab = SimpleBPE()
        vi_vocab = SimpleBPE()
    en_vocab.load(en_path)
    vi_vocab.load(vi_path)

    model = Transformer(
        src_vocab_size=cfg["src_vocab_size"],
        tgt_vocab_size=cfg["tgt_vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_encoder_layers=cfg["n_encoder_layers"],
        n_decoder_layers=cfg["n_decoder_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_len = cfg.get("max_seq_len", 50)

    print()
    print("Interactive translation. Type English, get Vietnamese.")
    print("Blank line or Ctrl-D to exit.")
    print()
    while True:
        try:
            text = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        vi_pred = generate_translation(model, text, en_vocab, vi_vocab, max_len=max_len, device=device)
        print(f"VI> {vi_pred}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Transformer on Vietnamese-English translation")
    parser.add_argument("--demo", action="store_true", help="Use small demo dataset")
    parser.add_argument("--full", action="store_true", help="Use full IWSLT dataset")
    parser.add_argument("--infer", action="store_true",
                        help="Skip training; load saved model + tokenizers for interactive translation")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of encoder/decoder layers")
    parser.add_argument("--d-ff", type=int, default=256, help="Feed-forward dimension")
    parser.add_argument("--max-len", type=int, default=50, help="Max sequence length")
    args = parser.parse_args()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 8 + "TRANSFORMER FROM SCRATCH - LESSON 8" + " " * 28 + "║")
    print("║" + " " * 10 + "IWSLT Vietnamese-English Translation" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    if args.infer:
        infer_loop()
        return

    # Configuration
    BATCH_SIZE = args.batch_size
    D_MODEL = args.d_model
    N_HEADS = args.n_heads
    N_ENCODER_LAYERS = args.n_layers
    N_DECODER_LAYERS = args.n_layers
    D_FF = args.d_ff
    DROPOUT = 0.1
    NUM_EPOCHS = args.epochs
    MAX_SEQ_LEN = args.max_len
    VOCAB_SIZE = args.vocab_size
    LEARNING_RATE = 0.001

    # Device - prefer MPS (Apple Silicon GPU) if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU only)")
    print()

    # ================================================================
    # DATASET
    # ================================================================
    print("Step 1: Loading dataset...")

    if args.demo or not args.full:
        train_pairs, val_pairs = load_demo_data()
    else:
        train_pairs, val_pairs = load_full_data()

    # Calculate warmup steps based on dataset size
    # For demo: use 0 warmup steps (start learning immediately with cosine decay)
    # For full: ~4166 steps/epoch (133K/32), so use 4000 steps
    WARMUP_STEPS = 0 if args.demo else 4000

    en_texts_train = [p[0] for p in train_pairs]
    vi_texts_train = [p[1] for p in train_pairs]
    en_texts_val = [p[0] for p in val_pairs]
    vi_texts_val = [p[1] for p in val_pairs]

    print(f"  English sentences: {len(en_texts_train):,}")
    print(f"  Vietnamese sentences: {len(vi_texts_train):,}")
    print()

    # ================================================================
    # TOKENIZER
    # ================================================================
    print("Step 2: Training tokenizers...")
    print()

    if HAS_TOKENIZERS:
        print("  Using HuggingFace tokenizers library")
        en_vocab = HuggingFaceBPE(vocab_size=VOCAB_SIZE)
        vi_vocab = HuggingFaceBPE(vocab_size=VOCAB_SIZE)

        # Write texts to temp files for training
        temp_en_path = "/tmp/train.en.txt"
        temp_vi_path = "/tmp/train.vi.txt"

        with open(temp_en_path, "w", encoding="utf-8") as f:
            for text in en_texts_train:
                f.write(text + "\n")
        with open(temp_vi_path, "w", encoding="utf-8") as f:
            for text in vi_texts_train:
                f.write(text + "\n")

        print("  Training English tokenizer...")
        en_vocab.train([temp_en_path])

        print("\n  Training Vietnamese tokenizer...")
        vi_vocab.train([temp_vi_path])

        # Clean up
        os.remove(temp_en_path)
        os.remove(temp_vi_path)
    else:
        print("  Using simple BPE implementation (install 'tokenizers' for better results)")
        print("  Training English tokenizer...")
        en_vocab = SimpleBPE(vocab_size=VOCAB_SIZE)
        en_vocab.train(en_texts_train)
        print(f"  English vocab size: {en_vocab.get_vocab_size()}")

        print("\n  Training Vietnamese tokenizer...")
        vi_vocab = SimpleBPE(vocab_size=VOCAB_SIZE)
        vi_vocab.train(vi_texts_train)
        print(f"  Vietnamese vocab size: {vi_vocab.get_vocab_size()}")

    # Persist tokenizers so --infer can reload them without retraining.
    en_tokenizer_path, vi_tokenizer_path = tokenizer_paths()
    en_vocab.save(en_tokenizer_path)
    vi_vocab.save(vi_tokenizer_path)
    print(f"  Saved tokenizers: {en_tokenizer_path}, {vi_tokenizer_path}")

    print()

    # ================================================================
    # CREATE DATASET
    # ================================================================
    print("Step 3: Creating datasets...")
    train_dataset = TranslationDataset(
        en_texts_train, vi_texts_train,
        en_vocab, vi_vocab, max_len=MAX_SEQ_LEN
    )
    val_dataset = TranslationDataset(
        en_texts_val, vi_texts_val,
        en_vocab, vi_vocab, max_len=MAX_SEQ_LEN
    )
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print()

    # ================================================================
    # CREATE MODEL
    # ================================================================
    print("Step 4: Creating model...")
    model = Transformer(
        src_vocab_size=en_vocab.get_vocab_size(),
        tgt_vocab_size=vi_vocab.get_vocab_size(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(0.9, 0.98), eps=1e-9)

    # Training configuration
    steps_per_epoch = len(train_loader)
    total_steps = NUM_EPOCHS * steps_per_epoch
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max sequence length: {MAX_SEQ_LEN}")
    print()

    current_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'='*70}")

        train_loss, train_acc, current_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            current_step, total_steps, WARMUP_STEPS, LEARNING_RATE,
            epoch, NUM_EPOCHS
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch, NUM_EPOCHS)

        current_lr = optimizer.param_groups[0]["lr"]
        perplexity = math.exp(min(train_loss, 100))

        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS} SUMMARY")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Learning Rate: {current_lr:.8f}")
        print()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_type": "huggingface" if HAS_TOKENIZERS else "simple",
                "model_config": {
                    "src_vocab_size": en_vocab.get_vocab_size(),
                    "tgt_vocab_size": vi_vocab.get_vocab_size(),
                    "d_model": D_MODEL,
                    "n_heads": N_HEADS,
                    "n_encoder_layers": N_ENCODER_LAYERS,
                    "n_decoder_layers": N_DECODER_LAYERS,
                    "d_ff": D_FF,
                    "dropout": DROPOUT,
                    "max_seq_len": MAX_SEQ_LEN,
                },
                "epoch": epoch,
            }, CHECKPOINT_PATH)
            print(f"  ✓ Saved best model (val loss: {val_loss:.4f})")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print()

    # Load best model
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test translations
    print("TESTING TRANSLATIONS")
    print("-" * 70)
    print()

    test_sentences = [
        "hello",
        "i am happy",
        "the cat is small",
        "good morning",
        "thank you",
    ]

    for eng in test_sentences:
        vi_pred = generate_translation(model, eng, en_vocab, vi_vocab, device=device)
        print(f"  English:    '{eng}'")
        print(f"  Predicted:  '{vi_pred}'")
        print()

    print("=" * 70)
    print("WHAT WE LEARNED:")
    print("-" * 70)
    print("✓ HuggingFace Tokenizers: Production-quality BPE tokenization")
    print("✓ Real Dataset: IWSLT Vietnamese-English corpus (~133K pairs)")
    print("✓ Learning Rate Schedule: Warmup + cosine decay")
    print("✓ Teacher Forcing: Feed correct tokens during training")
    print("✓ Greedy Decoding: Pick highest probability token")
    print("✓ Gradient Clipping: Prevent exploding gradients")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()