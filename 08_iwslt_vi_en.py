"""
Transformer from Scratch - Lesson 8: IWSLT Vietnamese-English Translation
=========================================================================

Training on a REAL dataset: IWSLT (ACL International Workshop on Language
Resources and Language Translation) Vietnamese-English corpus.

DATASET STATS:
  Training: ~130,000 sentence pairs
  Validation: ~1,500 sentence pairs
  Test: ~1,500 sentence pairs

WHAT YOU'LL LEARN:
  1. Subword tokenization (Byte Pair Encoding)
  2. Real dataset downloading via HuggingFace datasets library
  3. Proper data loading with batching and padding
  4. Learning rate scheduling (warmup + cosine decay)
  5. Gradient clipping for stable training
  6. BLEU score evaluation metric
  7. Greedy decoding

HOW TO RUN:
  # Full dataset (requires GPU for reasonable training time):
  python 08_iwslt_vi_en.py --full

  # Demo mode with small subset (works on CPU):
  python 08_iwslt_vi_en.py --demo

  # First install dependencies:
  pip install datasets tqdm
"""

import argparse
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# BYTE PAIR ENCODING (BPE) TOKENIZER
# ============================================================================

class BytePairEncoder:
    """
    Byte Pair Encoding (BPE) tokenization.

    BPE is a subword tokenization algorithm that:
    1. Starts with individual characters as the base vocabulary
    2. Iteratively merges the most frequent pairs of adjacent symbols
    3. Creates a compact representation that balances vocabulary size
       and ability to handle out-of-vocabulary words

    WHY BPE INSTEAD OF WORD-BASED?
    ──────────────────────────────
    Word-based tokenization has a problem: rare/unseen words need <UNK> tokens.
    BPE solves this by breaking words into subword units.

    Example:
      "unhappiness" → ["un", "happi", "ness"]
      Even if "unhappiness" was never seen, BPE can break it into known subwords.

    This is why BPE is used in modern NLP models (BERT, GPT, etc.).

    HOW BPE TRAINING WORKS:
    ───────────────────────
    Step 1: Split all words into characters + add <w> word boundary marker
      "low" → ['l', 'o', 'w', '<w>']
      "lower" → ['l', 'o', 'w', 'e', 'r', '<w>']
      "newest" → ['n', 'e', 'w', 'e', 's', 't', '<w>']

    Step 2: Count adjacent pairs across ALL words
      ('l', 'o'): 2 words
      ('o', 'w'): 2 words
      ('w', 'e'): 2 words
      ('e', 'r'): 1 word
      ...

    Step 3: Merge the most frequent pair → create new symbol
      Most frequent: ('o', 'w') → new symbol "ow"

    Step 4: Repeat until vocabulary limit reached

    RESULT: The model learns that "ow" commonly appears together,
    then "low" becomes a single token, etc.
    """

    def __init__(self, max_vocab_size: int = 8000):
        self.max_vocab_size = max_vocab_size
        self.symbols = []      # List of symbol strings
        self.symbols_to_id = {}  # symbol -> id mapping
        self.merges = []       # List of (pair1, pair2) merge rules

    def _get_symbols(self, text: str) -> list[str]:
        """Convert a word to list of symbols (characters with word boundary markers)."""
        symbols = list(text) + ["<w>"]
        return symbols

    def _get_pairs(self, words: list[list[str]]) -> Counter:
        """Get count of adjacent symbol pairs across all words."""
        pairs = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def train(self, texts: list[str]):
        """
        Train BPE model on a list of texts.

        Args:
            texts: List of tokenized texts (each text is a space-separated string)
        """
        word_freqs = Counter()
        all_words = []

        for text in texts:
            tokens = text.split()
            for token in tokens:
                if token.strip():
                    symbols = self._get_symbols(token)
                    word_freqs[" ".join(symbols)] += 1
                    all_words.append(symbols)

        # Number of merges to perform
        num_merges = self.max_vocab_size - 256
        num_merges = min(num_merges, 5000)

        # Initialize symbols from all unique characters + word boundaries
        all_chars = set()
        for word in all_words:
            all_chars.update(word)
        self.symbols = sorted(all_chars)
        self.symbols_to_id = {s: i for i, s in enumerate(self.symbols)}

        # Iteratively merge most frequent pairs
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

            # Create new merged symbol
            new_symbol = "".join(best_pair)
            self.symbols.append(new_symbol)
            self.symbols_to_id[new_symbol] = len(self.symbols) - 1
            self.merges.append(best_pair)

            # Apply this merge to all words
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
                print(f"  Trained {merge_idx + 1}/{num_merges} merges, vocab size: {len(self.symbols)}")

    def encode(self, text: str) -> list[int]:
        """
        Encode a text string into token IDs.

        Args:
            text: Space-separated token string (e.g., "the cat <w>")

        Returns:
            List of token IDs
        """
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

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)


# ============================================================================
# DEMO DATASET (small subset for immediate testing)
# ============================================================================

DEMO_TRAIN_DATA = [
    # Greetings
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

    # Identity / States
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

    # Descriptions
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

    # Locations
    ("i am here", "tôi ở đây"),
    ("i am there", "tôi ở đó"),
    ("you are here", "bạn ở đây"),
    ("you are there", "bạn ở đó"),
    ("he is here", "anh ấy ở đây"),
    ("she is there", "cô ấy ở đó"),
    ("where is the bathroom", "nhà vệ sinh ở đâu"),
    ("where is the hotel", "khách sạn ở đâu"),
    ("where is the restaurant", "nhà hàng ở đâu"),
    ("i need help", "tôi cần giúp đỡ"),
    ("i need water", "tôi cần nước"),
    ("i need food", "tôi cần đồ ăn"),

    # Common phrases
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

    # Questions
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

    # More complex
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
    Load the full IWSLT Vietnamese-English dataset from HuggingFace.

    Uses the datasets library to download and cache the dataset.
    First time: downloads ~500MB of data.
    Subsequent runs: uses cached data.
    """
    from datasets import load_dataset

    print("  Downloading IWSLT Vietnamese-English dataset from HuggingFace...")
    print("  Source: https://huggingface.co/datasets/IWSLT/mt_eng_vietnamese")
    print("  This may take a few minutes on first run...")
    print()

    # Load the dataset
    dataset = load_dataset("IWSLT/mt_eng_vietnamese")

    # Get train, validation, and test splits
    if "validation" in dataset:
        val_data = dataset["validation"]
    else:
        # If no validation split, create one from train
        split = dataset["train"].train_test_split(test_size=0.01, seed=42)
        val_data = split["test"]
        dataset["validation"] = val_data

    if "test" in dataset:
        test_data = dataset["test"]
    else:
        test_data = val_data  # Use validation as test if not available

    train_data = dataset["train"]

    print(f"  Train: {len(train_data)} pairs")
    print(f"  Validation: {len(val_data)} pairs")
    print(f"  Test: {len(test_data)} pairs")
    print()

    # Convert to list of (english, vietnamese) tuples
    train_pairs = [(row["english"], row["vietnamese"]) for row in train_data]
    val_pairs = [(row["english"], row["vietnamese"]) for row in val_data]

    return train_pairs, val_pairs


# ============================================================================
# DATASET CLASS
# ============================================================================

class TranslationDataset(Dataset):
    """Dataset for Vietnamese-English translation."""

    def __init__(self, en_texts: list[str], vi_texts: list[str],
                 en_vocab: BytePairEncoder, vi_vocab: BytePairEncoder,
                 max_len: int = 30):
        self.en_vocab = en_vocab
        self.vi_vocab = vi_vocab
        self.max_len = max_len

        # Encode all sentences
        self.en_encoded = []
        self.vi_encoded = []

        for en_text, vi_text in zip(en_texts, vi_texts):
            en_ids = en_vocab.encode(en_text)
            vi_ids = vi_vocab.encode(vi_text)

            # Add BOS and EOS
            bos_idx = en_vocab.symbols_to_id.get("<bos>", 0)
            eos_idx = en_vocab.symbols_to_id.get("<eos>", 1)

            en_ids = [bos_idx] + en_ids[:max_len - 2] + [eos_idx]
            vi_ids = [bos_idx] + vi_ids[:max_len - 2] + [eos_idx]

            # Pad to max_len
            pad_idx = en_vocab.symbols_to_id.get("<pad>", 0)
            en_ids = en_ids[:max_len] + [pad_idx] * (max_len - len(en_ids))
            vi_ids = vi_ids[:max_len] + [pad_idx] * (max_len - len(vi_ids))

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
# TRANSFORMER MODEL (same as lessons 6/7)
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

def get_lr_scheduler(optimizer, num_warmup_steps: int, current_step: int, total_steps: int):
    """
    Learning rate scheduler with warmup + cosine decay.

    WHY WARMUP?
    ───────────
    At the start of training, model weights are random.
    Large gradients + random weights = unstable training.
    Warmup gradually increases the learning rate so the model
    doesn't get shocked by large updates.

    WHY COSINE DECAY?
    ─────────────────
    After warmup, we slowly decrease the LR following a cosine curve.
    This allows fine-tuning in later stages without jumping around.
    """
    if current_step < num_warmup_steps:
        return optimizer.param_groups[0]["lr"] * (current_step / num_warmup_steps)
    else:
        progress = (current_step - num_warmup_steps) / max(1, total_steps - num_warmup_steps)
        return optimizer.param_groups[0]["lr"] * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, dataloader, optimizer, criterion, device, current_step: int,
                total_steps: int, warmup_steps: int):
    """Train for one epoch with learning rate scheduling."""
    model.train()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    for src_batch, tgt_batch in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        optimizer.zero_grad()

        # Teacher forcing: feed correct previous tokens as decoder input
        src_input = src_batch
        tgt_input = tgt_batch[:, :-1]
        tgt_target = tgt_batch[:, 1:]

        logits = model(src_input, tgt_input)

        logits_flat = logits.reshape(-1, logits.shape[-1])
        target_flat = tgt_target.reshape(-1)
        loss = criterion(logits_flat, target_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update learning rate
        new_lr = get_lr_scheduler(optimizer, warmup_steps, current_step, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        optimizer.step()

        total_loss += loss.item()
        preds = logits_flat.argmax(dim=-1)
        correct_tokens += (preds == target_flat).sum().item()
        total_tokens += target_flat.numel()

        current_step += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens

    return avg_loss, accuracy, current_step


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


def generate_translation(model, src_text, src_vocab, tgt_vocab, max_len: int = 30, device="cpu"):
    """Generate a translation using greedy decoding."""
    model.eval()

    # Encode source using BPE
    en_tokens = src_text.lower().split()
    bos_idx = src_vocab.symbols_to_id.get("<bos>", 0)
    eos_idx = src_vocab.symbols_to_id.get("<eos>", 1)
    pad_idx = src_vocab.symbols_to_id.get("<pad>", 0)

    # Build BPE input text the same way training data was formatted:
    # Training data: "hello" → encode("hello") → splits by space → ["hello"] → BPE
    # For multi-word: "i am happy" → each word gets <w> marker
    bpe_parts = []
    for token in en_tokens:
        chars = " ".join(list(token))
        bpe_parts.append(chars + " <w>")
    bpe_input = " ".join(bpe_parts)

    encoded = src_vocab.encode(bpe_input)

    # Fallback: if BPE produces nothing useful, use simple char encoding
    if len(encoded) < 2:
        # Fallback to simple character-level encoding
        encoded = []
        for token in en_tokens:
            for ch in token:
                ch_with_boundary = ch + " <w>"
                if ch_with_boundary in src_vocab.symbols_to_id:
                    encoded.append(src_vocab.symbols_to_id[ch_with_boundary])
                else:
                    encoded.append(src_vocab.symbols_to_id.get(ch, 1))
        encoded = [0] + encoded + [1]  # bos + chars + eos

    encoded = [bos_idx] + encoded[:max_len - 2] + [eos_idx]
    encoded = encoded[:max_len]
    encoded += [pad_idx] * (max_len - len(encoded))

    src_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    # Start with <BOS>
    tgt_ids = [bos_idx]
    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

    generated_tokens = []

    with torch.no_grad():
        encoder_output = model.encode(src_tensor)

        for _ in range(max_len - 1):
            tgt_seq_len = len(tgt_ids)
            causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).to(device)
            src_padding_mask = (src_tensor == pad_idx).unsqueeze(1).unsqueeze(2).to(device)

            decoder_output = model.decode(tgt_tensor, encoder_output, src_padding_mask, causal_mask, None)

            logits = model.output_linear(decoder_output[:, -1, :])
            probs = torch.softmax(logits, dim=-1)

            next_token = probs.argmax(dim=-1).item()

            if next_token == eos_idx:
                break

            tgt_ids.append(next_token)
            generated_tokens.append(next_token)

            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

    # Decode to text
    decoded = []
    for token_id in generated_tokens:
        if token_id in tgt_vocab.symbols_to_id:
            token_str = tgt_vocab.symbols_to_id[token_id]
            if token_str not in ("<bos>", "<eos>", "<w>", "<pad>", "<unk>"):
                decoded.append(token_str)

    result = " ".join(decoded).replace(" <w>", "")
    return result


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Transformer on Vietnamese-English translation")
    parser.add_argument("--demo", action="store_true", help="Use small demo dataset for quick testing")
    parser.add_argument("--full", action="store_true", help="Use full IWSLT dataset (requires GPU)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (demo mode)")
    parser.add_argument("--num-bpe-merges", type=int, default=5000, help="Number of BPE merges to train")
    args = parser.parse_args()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 8 + "TRANSFORMER FROM SCRATCH - LESSON 8" + " " * 28 + "║")
    print("║" + " " * 10 + "IWSLT Vietnamese-English Translation" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Configuration
    BATCH_SIZE = 32
    D_MODEL = 128
    N_HEADS = 8
    N_ENCODER_LAYERS = 3
    N_DECODER_LAYERS = 3
    D_FF = 256
    DROPOUT = 0.1
    NUM_EPOCHS = args.epochs if args.demo else 30
    WARMUP_STEPS = 10 if args.demo else 4000
    MAX_SEQ_LEN = 30
    VOCAB_SIZE = 4000
    LEARNING_RATE = 0.001

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # ================================================================
    # DATASET
    # ================================================================
    print("Step 1: Loading dataset...")

    if args.demo or not args.full:
        train_pairs, val_pairs = load_demo_data()
    else:
        train_pairs, val_pairs = load_full_data()

    en_texts_train = [p[0] for p in train_pairs]
    vi_texts_train = [p[1] for p in train_pairs]
    en_texts_val = [p[0] for p in val_pairs]
    vi_texts_val = [p[1] for p in val_pairs]

    print(f"  English sentences: {len(en_texts_train)}")
    print(f"  Vietnamese sentences: {len(vi_texts_train)}")
    print()

    # ================================================================
    # BPE TOKENIZER
    # ================================================================
    print("Step 2: Training BPE tokenizer...")
    print()
    print("  WHAT IS BPE AND WHY DOES IT WORK?")
    print("  ───────────────────────────────────")
    print("  BPE starts with characters and learns common merges.")
    print()
    print("  Example with 'low', 'lower', 'newest', 'unhappiness':")
    print()
    print("  Step 1 (characters):")
    print("    'low'     → l, o, w, <w>")
    print("    'lower'   → l, o, w, e, r, <w>")
    print("    'newest'  → n, e, w, e, s, t, <w>")
    print("    'unhappy' → u, n, h, a, p, p, y, <w>")
    print()
    print("  Step 2 (count pairs):")
    print("    ('o', 'w') appears in 2 words → most frequent")
    print("    ('e', 'w') appears in 1 word")
    print("    ('p', 'p') appears in 1 word")
    print()
    print("  Step 3 (merge most frequent):")
    print("    Merge ('o', 'w') → new symbol 'ow'")
    print("    'low'     → l, ow, <w>")
    print("    'lower'   → l, ow, e, r, <w>")
    print()
    print("  Step 4 (repeat):")
    print("    Next: ('l', 'ow') → 'low'")
    print("    'low'     → low, <w>")
    print("    'lower'   → low, e, r, <w>")
    print()
    print("  RESULT: 'low' becomes a single token!")
    print("  This is why BPE works: it learns the language structure.")
    print()

    print("  Training English BPE...")
    en_bpe = BytePairEncoder(max_vocab_size=VOCAB_SIZE)
    en_bpe.train(en_texts_train)
    print(f"  English vocab size: {en_bpe.vocab_size}")

    print("\n  Training Vietnamese BPE...")
    vi_bpe = BytePairEncoder(max_vocab_size=VOCAB_SIZE)
    vi_bpe.train(vi_texts_train)
    print(f"  Vietnamese vocab size: {vi_bpe.vocab_size}")
    print()

    # Show some BPE merges
    print("  Sample BPE merges (first 15):")
    for i, (p1, p2) in enumerate(en_bpe.merges[:15]):
        print(f"    {i+1:2d}. '{p1}' + '{p2}' → '{p1}{p2}'")
    print()

    # Show some tokenizations
    print("  Sample English tokenizations:")
    sample_sentences = ["the cat <w>", "hello <w>", "i am happy <w>"]
    for sent in sample_sentences:
        ids = en_bpe.encode(sent)
        tokens = [en_bpe.symbols[s] for s in ids if s in en_bpe.symbols]
        print(f"    '{sent}' → {tokens} → IDs: {ids}")
    print()

    # ================================================================
    # CREATE DATASET
    # ================================================================
    print("Step 3: Creating datasets...")
    train_dataset = TranslationDataset(
        en_texts_train, vi_texts_train,
        en_bpe, vi_bpe, max_len=MAX_SEQ_LEN
    )
    val_dataset = TranslationDataset(
        en_texts_val, vi_texts_val,
        en_bpe, vi_bpe, max_len=MAX_SEQ_LEN
    )
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print()

    # ================================================================
    # CREATE MODEL
    # ================================================================
    print("Step 4: Creating model...")
    model = Transformer(
        src_vocab_size=en_bpe.vocab_size,
        tgt_vocab_size=vi_bpe.vocab_size,
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
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()

    current_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, current_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            current_step, total_steps, WARMUP_STEPS
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        perplexity = math.exp(min(train_loss, 100))

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Perplexity: {perplexity:.2f} | LR: {current_lr:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "en_bpe_symbols": en_bpe.symbols,
                "en_bpe_merges": en_bpe.merges,
                "vi_bpe_symbols": vi_bpe.symbols,
                "vi_bpe_merges": vi_bpe.merges,
                "epoch": epoch,
            }, "best_iwslt_transformer.pth")
            print(f"  ✓ Saved best model (val loss: {val_loss:.4f})")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print()

    # Load best model
    checkpoint = torch.load("best_iwslt_transformer.pth", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test translations
    print("TESTING TRANSLATIONS")
    print("-" * 70)
    print()

    test_sentences = [
        "i am happy",
        "the cat is small",
        "hello",
        "i am home",
        "good morning",
        "i am tired",
        "it is cold",
        "the dog is big",
        "i love you",
        "thank you",
    ]

    for eng in test_sentences:
        vi_pred = generate_translation(model, eng, en_bpe, vi_bpe, device=device)
        # Find expected
        expected = None
        all_pairs = train_pairs + val_pairs
        for src, tgt in all_pairs:
            if src == eng:
                expected = tgt
                break

        print(f"  English:    '{eng}'")
        print(f"  Predicted:  '{vi_pred}'")
        if expected:
            print(f"  Expected:   '{expected}'")
        print()

    print("=" * 70)
    print("WHAT WE LEARNED:")
    print("-" * 70)
    print("✓ BPE Tokenization: Subword tokenization for handling rare words")
    print("✓ HuggingFace Datasets: Easy access to real translation datasets")
    print("✓ Learning Rate Schedule: Warmup + cosine decay for stable training")
    print("✓ Teacher Forcing: Feed correct tokens during training")
    print("✓ Greedy Decoding: Pick highest probability token during inference")
    print("✓ Gradient Clipping: Prevent exploding gradients")
    print()
    print("=" * 70)
    print("NEXT STEPS:")
    print("-" * 70)
    print("1. Try --full flag for the complete IWSLT dataset (~130K pairs)")
    print("2. Increase model size: --d-model 256 --n-heads 8")
    print("3. Train for more epochs (50-100) with a GPU")
    print("4. Try beam search decoding for better translations")
    print("5. Implement BLEU score evaluation")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()