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

        Example:
            >>> tokenizer = HuggingFaceBPE(vocab_size=1000)
            >>> # Create a sample text file
            >>> with open("sample.txt", "w") as f:
            ...     f.write("hello world\\nhello there\\ngoodbye world")
            >>> tokenizer.train(["sample.txt"])
            >>> # Now you can encode text
            >>> ids = tokenizer.encode("hello world")
            >>> # ids might be: [15, 234, 567, 89]  # token IDs
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
        """
        Encode text to token IDs.

        Args:
            text: Input text string to encode

        Returns:
            List of token IDs

        Example:
            >>> # Assuming trained tokenizer
            >>> tokenizer.encode("hello world")
            [15, 234, 567, 89]
            >>> tokenizer.encode("I am happy")
            [2, 45, 78, 123]
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet")
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs to decode

        Returns:
            Decoded text string

        Example:
            >>> # Assuming trained tokenizer
            >>> tokenizer.decode([15, 234, 567, 89])
            'hello world'
            >>> tokenizer.decode([2, 45, 78, 123])
            'i am happy'
        """
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
        """
        Split text into character-level symbols with word boundary marker.

        Args:
            text: Input text string

        Returns:
            List of symbols (characters + <w> word boundary)

        Example:
            >>> tokenizer = SimpleBPE()
            >>> tokenizer._get_symbols("cat")
            ['c', 'a', 't', '<w>']
        """
        symbols = list(text) + ["<w>"]
        return symbols

    def _get_pairs(self, words: list[list[str]]) -> Counter:
        """
        Count adjacent symbol pairs across all words.

        Args:
            words: List of words, each word is a list of symbols

        Returns:
            Counter of (symbol1, symbol2) pairs

        Example:
            >>> tokenizer = SimpleBPE()
            >>> words = [['c', 'a', 't', '<w>'], ['d', 'o', 'g', '<w>']]
            >>> pairs = tokenizer._get_pairs(words)
            >>> ('c', 'a') in pairs
            True
        """
        pairs = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def train(self, texts: list[str]):
        """
        Train BPE tokenizer on text data.

        Args:
            texts: List of text strings to train on

        Example:
            >>> tokenizer = SimpleBPE(vocab_size=500)
            >>> texts = ["hello world", "hello there", "goodbye world"]
            >>> tokenizer.train(texts)
            >>> # Now encode text
            >>> ids = tokenizer.encode("hello")
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
        """
        Encode text to token IDs.

        Args:
            text: Input text string to encode

        Returns:
            List of token IDs

        Example:
            >>> # After training: tokenizer.train(["hello world", "goodbye"])
            >>> tokenizer.encode("hello world")
            [45, 78, 123]  # Example IDs
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

    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs to decode

        Returns:
            Decoded text string

        Example:
            >>> # After training
            >>> tokenizer.decode([45, 78, 123])
            'hello world'
        """
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
    """
    Load the small demo dataset for quick testing.

    Returns:
        Tuple of (training_pairs, validation_pairs)

    Example:
        >>> train_data, val_data = load_demo_data()
        >>> len(train_data)
        91
        >>> train_data[0]
        ('hello', 'xin chào')
        >>> val_data[0]
        ('good night', 'chúc ngủ ngon')
    """
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

    Returns:
        Tuple of (training_pairs, validation_pairs)

    Example:
        >>> # First ensure data files exist in data/ directory
        >>> train_data, val_data = load_full_data()
        >>> len(train_data)  # ~133,000 pairs
        133317
        >>> len(val_data)    # ~1,500 pairs
        1553
        >>> # Each pair is (English_sentence, Vietnamese_sentence)
        >>> train_data[0]
        ('I would like to know your name.', 'Tôi muốn biết tên của bạn.')
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
    """
    Dataset for Vietnamese-English translation.

    This class handles tokenization, adding special tokens (BOS, EOS),
    and padding sequences to a fixed length.

    Special Token IDs:
        - <pad> = 0: Padding token
        - <unk> = 1: Unknown token
        - <bos> = 2: Beginning of sequence
        - <eos> = 3: End of sequence

    Example:
        >>> # Assume we have trained vocabularies
        >>> en_texts = ["hello world", "good morning"]
        >>> vi_texts = ["xin chào thế giới", "chào buổi sáng"]
        >>> dataset = TranslationDataset(en_texts, vi_texts, en_vocab, vi_vocab, max_len=10)
        >>> len(dataset)
        2
        >>> # Get a sample - returns (english_ids, vietnamese_ids) tensors
        >>> en_ids, vi_ids = dataset[0]
        >>> en_ids.shape
        torch.Size([10])
        >>> # Sequence looks like: [<bos>, token_ids..., <eos>, <pad>, <pad>...]
    """

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
        """
        Return the number of samples in the dataset.

        Example:
            >>> len(dataset)
            1000
        """
        return len(self.en_encoded)

    def __getitem__(self, idx: int):
        """
        Get a single training sample by index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (english_tensor, vietnamese_tensor), each of shape (max_len,)

        Example:
            >>> en_tensor, vi_tensor = dataset[5]
            >>> en_tensor.shape
            torch.Size([50])
            >>> en_tensor[0].item()  # Should be <bos> token ID
            2
        """
        return (
            torch.tensor(self.en_encoded[idx], dtype=torch.long),
            torch.tensor(self.vi_encoded[idx], dtype=torch.long),
        )


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class TokenEmbedding(nn.Module):
    """
    Token embedding scaled by sqrt(d_model), per Vaswani et al. 2017.

    WHY THE EXPLICIT INIT (this is the important bit):

    PyTorch's default init for `nn.Embedding` is N(0, 1). In forward() we
    multiply by sqrt(d_model), so without a custom init the embeddings
    emerging from this module have stddev sqrt(d_model) at step 0 —
    ~22.6 for d_model=512.

    Meanwhile the sinusoidal positional encoding added right after has
    values in [-1, 1]. So at initialization, token identity is roughly
    20× louder than position, and the model effectively can't see order
    until many gradient steps rescale things. That's what "slow learning
    at large d_model" usually looks like.

    The paper's assumption is that the raw embedding weights start at
    ~N(0, 1/sqrt(d_model)), so that the `* sqrt(d_model)` scaling in
    forward() brings them back to ~N(0, 1) — matching the positional
    encoding scale. We init explicitly here to make that assumption true.

    Example:
        >>> vocab_size, d_model = 100, 64
        >>> embed = TokenEmbedding(vocab_size, d_model)
        >>> # Input: batch of 2 sequences, each with 5 token IDs
        >>> x = torch.tensor([[1, 5, 10, 2, 0], [3, 8, 15, 2, 0]])
        >>> output = embed(x)
        >>> output.shape
        torch.Size([2, 5, 64])
        >>> # Output is scaled by sqrt(d_model) = 8
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Start embeddings at a scale that cancels the sqrt(d_model) factor
        # applied in forward(). See class docstring for the full reasoning.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, x):
        """
        Embed and scale token IDs.

        Args:
            x: Token IDs tensor of shape (batch_size, seq_len)

        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model), scaled by sqrt(d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in Vaswani et al. 2017.

    Adds positional information to token embeddings using sine and cosine
    functions of different frequencies. This allows the model to leverage
    sequence order since the transformer has no inherent notion of position.

    Why sinusoidal? It allows extrapolation to sequence lengths longer
    than trained, because the pattern is deterministic and unbounded.

    Example:
        >>> d_model = 64
        >>> pos_enc = PositionalEncoding(d_model, max_seq_len=100)
        >>> # Input: batch of 2 sequences with 10 tokens each
        >>> x = torch.randn(2, 10, d_model)  # e.g., from TokenEmbedding
        >>> output = pos_enc(x)
        >>> output.shape
        torch.Size([2, 10, 64])
        >>> # Positional encoding is added to input and dropout applied
    """

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
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added, same shape as input
        """
        return self.dropout(x + self.pe[:, :x.size(1), :])


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism (self-attention).

    This allows the model to attend to multiple positions simultaneously
    by splitting the embedding into multiple "heads" that each learn
    different attention patterns.

    How it works:
    1. Project input into Q (query), K (key), V (value) spaces
    2. Split into multiple heads for parallel attention
    3. Compute attention scores: Q · K^T / sqrt(d_k)
    4. Apply softmax and optional masking
    5. Weight values by attention scores
    6. Concatenate heads and project output

    Example:
        >>> d_model, n_heads = 64, 4
        >>> attn = MultiHeadAttention(d_model, n_heads)
        >>> # Self-attention: Q, K, V all from same source
        >>> x = torch.randn(2, 10, d_model)  # batch=2, seq_len=10
        >>> output, attn_weights = attn(x, x, x)
        >>> output.shape
        torch.Size([2, 10, 64])
        >>> attn_weights.shape  # attention distribution
        torch.Size([2, 4, 10, 10])  # (batch, heads, seq_len, seq_len)
    """

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
        """
        Split last dimension into multiple heads and transpose for attention.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor of shape (batch, n_heads, seq_len, d_k)
        """
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        """
        Merge heads back together after attention.

        Args:
            x: Tensor of shape (batch, n_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Compute multi-head attention.

        Args:
            Q: Query tensor of shape (batch, seq_len, d_model)
            K: Key tensor of shape (batch, seq_len, d_model)
            V: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional mask tensor (1=attend, 0=mask)

        Returns:
            Tuple of (output, attention_weights):
            - output: shape (batch, seq_len, d_model)
            - attention_weights: shape (batch, n_heads, seq_len, seq_len)

        How Masking Works in MultiHeadAttention:
        ────────────────────────────────────────
        The mask is applied to attention scores BEFORE softmax. Where mask=0,
        the score is set to -1e9 (negative infinity), making softmax output ~0.

        IMPORTANT: The scores matrix has shape (batch, n_heads, seq_len, seq_len).
        Each row i contains attention scores from query position i to ALL key positions.
        The mask operates on this seq_len × seq_len matrix per head.

        Code from forward():
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            # scores shape: (batch, n_heads, seq_len, seq_len)
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # Add head dimension
                # mask is now (batch, 1, 1, seq_len) for padding masks
                # or (batch, 1, seq_len, seq_len) for causal masks
                scores = scores.masked_fill(mask == 0, -1e9)
            attn = torch.softmax(scores, dim=-1)

        KEY INSIGHT: The mask broadcasts across the scores matrix.
        - For padding masks: mask columns corresponding to pad positions
        - For causal masks: mask upper triangle (future positions)

        ────────────────────────────────────────
        1. src_padding_mask in Encoder Self-Attention
        ────────────────────────────────────────
        Shape: (batch, 1, 1, src_len)
        After unsqueeze in create_masks: (batch, 1, 1, src_len)

        Example: batch=1, src_len=5, last 2 tokens are padding
        src = [tok0, tok1, tok2, pad, pad]
        src_padding_mask = [[[[1, 1, 1, 0, 0]]]]  # shape (1, 1, 1, 5)

        scores shape: (1, n_heads, 5, 5)
        For each head, the scores matrix looks like:
                    key0  key1  key2  key3  key4
            query0  [s00   s01   s02   s03   s04]
            query1  [s10   s11   s12   s13   s14]
            query2  [s20   s21   s22   s23   s24]
            query3  [s30   s31   s32   s33   s34]
            query4  [s40   s41   s42   s43   s44]

        After masked_fill with mask [1,1,1,0,0]:
        Columns 3,4 (key3, key4 = pad positions) get -1e9:
                    key0  key1  key2  key3   key4
            query0  [s00   s01   s02   -1e9   -1e9]
            query1  [s10   s11   s12   -1e9   -1e9]
            query2  [s20   s21   s22   -1e9   -1e9]
            query3  [s30   s31   s32   -1e9   -1e9]
            query4  [s40   s41   s42   -1e9   -1e9]

        After softmax (row by row):
                    key0   key1   key2   key3  key4
            query0  [0.25   0.35   0.40   0.0    0.0  ]
            query1  [0.30   0.30   0.40   0.0    0.0  ]
            ...     (each row sums to 1.0, pad columns are 0)

        ────────────────────────────────────────
        2. tgt_causal_mask in Decoder Self-Attention
        ────────────────────────────────────────
        Shape: (tgt_len, tgt_len) - lower triangular matrix
        This mask is created by: torch.tril(torch.ones(tgt_len, tgt_len))

        For tgt_len=5, the causal mask looks like:
                    pos0  pos1  pos2  pos3  pos4
            pos0    [1     0     0     0     0    ]  # pos0 sees only pos0
            pos1    [1     1     0     0     0    ]  # pos1 sees pos0,1
            pos2    [1     1     1     0     0    ]  # pos2 sees pos0,1,2
            pos3    [1     1     1     1     0    ]  # pos3 sees pos0,1,2,3
            pos4    [1     1     1     1     1    ]  # pos4 sees all

        In the decoder, when computing self-attention at position 2:
        - Position 2 can only attend to positions 0, 1, 2 (past and current)
        - Positions 3, 4 (future) are masked with -1e9
        - This prevents "cheating" by seeing future tokens during training

        scores shape: (batch, n_heads, tgt_len, tgt_len)
        After applying causal mask, upper triangle becomes -1e9:
                    pos0   pos1   pos2   pos3   pos4
            pos0    [s00   -1e9   -1e9   -1e9   -1e9  ]
            pos1    [s10    s11   -1e9   -1e9   -1e9  ]
            pos2    [s20    s21    s22   -1e9   -1e9  ]
            pos3    [s30    s31    s32    s33   -1e9  ]
            pos4    [s40    s41    s42    s43    s44  ]

        ────────────────────────────────────────
        3. Combined Mask in Decoder (causal + padding)
        ────────────────────────────────────────
        In Transformer.decode(), masks are combined:
            self_attn_mask = tgt_causal_mask.bool() & tgt_padding_mask.bool()

        Example: tgt_len=5, last 2 positions are padding
        tgt = [tok0, tok1, tok2, pad, pad]

        Causal mask:        Padding mask:       Combined:
        [1 0 0 0 0]         [1 1 1 0 0]         [1 0 0 0 0]
        [1 1 0 0 0]  AND    [1 1 1 0 0]  =      [1 1 0 0 0]
        [1 1 1 0 0]         [1 1 1 0 0]         [1 1 1 0 0]
        [1 1 1 1 0]         [1 1 1 0 0]         [1 1 1 0 0]
        [1 1 1 1 1]         [1 1 1 0 0]         [1 1 1 0 0]

        Result: Each position can attend to real past tokens only.
        - Future positions (causal) are masked
        - Padding positions are masked
        - Row 3 can only attend to pos0, pos1, pos2 (not pos3, pos4 which are pad)

        ────────────────────────────────────────
        4. src_padding_mask in Cross-Attention
        ────────────────────────────────────────
        Shape: (batch, 1, 1, src_len)
        Used when decoder attends to encoder output.

        Example: Decoder query at position 2 attending to encoder output
        Encoder processed: [tok0, tok1, tok2, pad, pad]
        src_padding_mask = [[[[1, 1, 1, 0, 0]]]]

        Cross-attention scores shape: (batch, n_heads, tgt_len, src_len)
        For each decoder position, attention to encoder pad positions is masked:
                    enc0   enc1   enc2   enc3   enc4
            dec0    [s00   s01    s02    -1e9   -1e9  ]
            dec1    [s10   s11    s12    -1e9   -1e9  ]
            dec2    [s20   s21    s22    -1e9   -1e9  ]
            ...

        Example:
            >>> # Self-attention with padding mask
            >>> d_model, n_heads = 64, 4
            >>> attn = MultiHeadAttention(d_model, n_heads)
            >>> x = torch.randn(2, 10, d_model)
            >>> # Mask last 2 positions as padding
            >>> mask = torch.ones(2, 1, 1, 10)
            >>> mask[:, :, :, 8:] = 0  # Last 2 are pad
            >>> output, attn_weights = attn(x, x, x, mask)
            >>> # Attention weights at masked positions will be ~0
            >>> attn_weights[0, 0, 0, 8:].sum()  # Attention to pad positions
            tensor(0.)
        """
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
    """
    Cross-attention mechanism for decoder to attend to encoder output.

    Unlike self-attention where Q, K, V come from the same source,
    cross-attention uses:
    - Q (query) from the decoder
    - K (key) and V (value) from the encoder output

    This allows the decoder to focus on relevant parts of the input
    sequence when generating each output token.

    Example:
        >>> d_model, n_heads = 64, 4
        >>> cross_attn = CrossAttention(d_model, n_heads)
        >>> # Decoder query (what the decoder is currently generating)
        >>> decoder_input = torch.randn(2, 8, d_model)   # batch=2, tgt_len=8
        >>> # Encoder output (encoded source sentence)
        >>> encoder_output = torch.randn(2, 12, d_model) # batch=2, src_len=12
        >>> output, attn_weights = cross_attn(decoder_input, encoder_output)
        >>> output.shape
        torch.Size([2, 8, 64])
        >>> attn_weights.shape  # shows which encoder positions decoder attends to
        torch.Size([2, 4, 8, 12])  # (batch, heads, tgt_len, src_len)
    """

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
        """
        Split last dimension into multiple heads and transpose for attention.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor of shape (batch, n_heads, seq_len, d_k)
        """
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        """
        Merge heads back together after attention.

        Args:
            x: Tensor of shape (batch, n_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, decoder_input, encoder_output, mask=None):
        """
        Compute cross-attention from decoder to encoder.

        Args:
            decoder_input: Query tensor from decoder, shape (batch, tgt_len, d_model)
            encoder_output: Key/Value tensor from encoder, shape (batch, src_len, d_model)
            mask: Optional mask tensor (1=attend, 0=mask)

        Returns:
            Tuple of (output, attention_weights):
            - output: shape (batch, tgt_len, d_model)
            - attention_weights: shape (batch, n_heads, tgt_len, src_len)
        """
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
    """
    Position-wise feed-forward network applied after attention.

    Each position is processed independently with the same two linear
    transformations with a ReLU activation in between:
        FFN(x) = max(0, x·W1 + b1)·W2 + b2

    The inner dimension (d_ff) is typically 4x larger than d_model,
    allowing the model to capture more complex patterns.

    Example:
        >>> d_model, d_ff = 64, 256
        >>> ffn = FeedForwardNetwork(d_model, d_ff)
        >>> x = torch.randn(2, 10, d_model)  # batch=2, seq_len=10
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([2, 10, 64])
        >>> # Each position is transformed independently
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Apply feed-forward transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of same shape (batch, seq_len, d_model)
        """
        return self.net(x)


class LayerNorm(nn.Module):
    """
    Layer normalization for stabilizing training.

    Unlike batch normalization, layer norm normalizes across features
    for each individual sample, making it suitable for sequence models
    and variable-length inputs.

    Formula: LayerNorm(x) = gamma * (x - mean) / (std + eps) + beta

    Where gamma and beta are learnable parameters.

    Example:
        >>> d_model = 64
        >>> norm = LayerNorm((d_model,))
        >>> x = torch.randn(2, 10, d_model)  # batch=2, seq_len=10
        >>> output = norm(x)
        >>> output.shape
        torch.Size([2, 10, 64])
        >>> # Each position is normalized independently
        >>> output.mean(dim=-1).mean()  # Should be close to 0
        tensor(0.)
    """

    def __init__(self, sizes: tuple[int, ...], eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(sizes))
        self.beta = nn.Parameter(torch.zeros(sizes))
        self.eps = eps

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape
        """
        mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True, unbiased=False)
        return ((x - mean) / (std + self.eps)) * self.gamma + self.beta


class EncoderLayer(nn.Module):
    """
    Single encoder layer from the Transformer model.

    Each encoder layer consists of:
    1. Multi-head self-attention mechanism
    2. Feed-forward network
    3. Layer normalization and residual connections

    The residual connections (x + sublayer(x)) help gradients flow
    through the network and make training more stable.

    Architecture:
        input → self-attention → norm → residual → FFN → norm → residual → output

    Example:
        >>> d_model, n_heads, d_ff = 64, 4, 128
        >>> encoder_layer = EncoderLayer(d_model, n_heads, d_ff)
        >>> # Input: batch of 2 sequences with 10 tokens each
        >>> x = torch.randn(2, 10, d_model)
        >>> output = encoder_layer(x)
        >>> output.shape
        torch.Size([2, 10, 64])
        >>> # With padding mask (1=keep, 0=mask)
        >>> mask = torch.tensor([[1,1,1,0,0], [1,1,1,1,0]]).unsqueeze(1).unsqueeze(2)
        >>> output = encoder_layer(x, mask)
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_padding_mask=None):
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            src_padding_mask: Optional mask for padding positions

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        attn_out, _ = self.self_attention(x, x, x, src_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer from the Transformer model.

    Each decoder layer consists of:
    1. Masked multi-head self-attention (causal mask prevents attending to future tokens)
    2. Cross-attention over encoder output
    3. Feed-forward network
    4. Layer normalization and residual connections

    The causal mask ensures that predictions for position i can only
    depend on positions < i (autoregressive property).

    Architecture:
        input → masked self-attention → norm → residual
              → cross-attention → norm → residual
              → FFN → norm → residual → output

    Example:
        >>> d_model, n_heads, d_ff = 64, 4, 128
        >>> decoder_layer = DecoderLayer(d_model, n_heads, d_ff)
        >>> # Decoder input (target sequence)
        >>> x = torch.randn(2, 8, d_model)   # batch=2, tgt_len=8
        >>> # Encoder output (source sequence)
        >>> encoder_output = torch.randn(2, 12, d_model)  # batch=2, src_len=12
        >>> # Causal mask (lower triangular)
        >>> causal_mask = torch.tril(torch.ones(8, 8))
        >>> output = decoder_layer(x, encoder_output, causal_mask)
        >>> output.shape
        torch.Size([2, 8, 64])
    """

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
        """
        Forward pass through decoder layer.

        Args:
            x: Decoder input tensor of shape (batch, tgt_len, d_model)
            encoder_output: Encoder output of shape (batch, src_len, d_model)
            causal_mask: Lower triangular mask to prevent attending to future
            padding_mask: Optional mask for padding positions

        Returns:
            Output tensor of shape (batch, tgt_len, d_model)
        """
        attn_out, _ = self.masked_attention(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        cross_out, _ = self.cross_attention(x, encoder_output, padding_mask)
        x = self.norm2(x + self.dropout(cross_out))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x


class Transformer(nn.Module):
    """
    Full Transformer model for sequence-to-sequence translation.

    Architecture overview:
    1. Source tokens → Token Embedding → Positional Encoding → Encoder stack
    2. Target tokens → Token Embedding → Positional Encoding → Decoder stack
    3. Decoder output → Linear projection → Vocabulary logits

    The encoder processes the entire source sequence in parallel, while
    the decoder generates the target sequence autoregressively (one token
    at a time, using causal masking).

    Model configuration (default):
        - d_model=128: Embedding dimension
        - n_heads=8: Number of attention heads
        - n_encoder_layers=3: Number of encoder layers
        - n_decoder_layers=3: Number of decoder layers
        - d_ff=256: Feed-forward hidden dimension (2x d_model)
        - dropout=0.1: Dropout rate

    Example:
        >>> src_vocab_size, tgt_vocab_size = 1000, 1000
        >>> model = Transformer(
        ...     src_vocab_size=src_vocab_size,
        ...     tgt_vocab_size=tgt_vocab_size,
        ...     d_model=64,
        ...     n_heads=4,
        ...     n_encoder_layers=2,
        ...     n_decoder_layers=2,
        ...     d_ff=128
        ... )
        >>> # Input: batch of 2 sequences
        >>> src = torch.randint(0, src_vocab_size, (2, 10))  # source: 10 tokens
        >>> tgt = torch.randint(0, tgt_vocab_size, (2, 8))   # target: 8 tokens
        >>> output = model(src, tgt)
        >>> output.shape  # (batch, tgt_seq_len, tgt_vocab_size)
        torch.Size([2, 8, 1000])
        >>> # Get predicted token IDs
        >>> predictions = output.argmax(dim=-1)
        >>> predictions.shape
        torch.Size([2, 8])
    """

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

        self._init_linear_weights()

    def _init_linear_weights(self):
        """Xavier-uniform init for every nn.Linear in the model.

        PyTorch's default init for Linear is kaiming_uniform with fan_in,
        which is tuned for ReLU-style networks. The Transformer's attention
        blocks compute symmetric Q·Kᵀ dot products and then softmax — that
        pathway benefits from Xavier (also called Glorot) uniform, which
        sets Var(W) = 2/(fan_in + fan_out). This keeps activation variance
        roughly constant across all 6 encoder + 6 decoder layers, so the
        deep stack doesn't blow up or collapse before training starts.

        Vaswani et al. 2017 use Xavier init throughout; it matters more as
        the model gets deeper. We only touch Linear weights and biases —
        embeddings are intentionally left alone because TokenEmbedding
        already sets its own scale-aware init.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def create_masks(self, src, tgt):
        """
        Create attention masks for encoder and decoder.

        Args:
            src: Source tensor of shape (batch, src_seq_len)
            tgt: Target tensor of shape (batch, tgt_seq_len)

        Returns:
            Tuple of (src_padding_mask, tgt_causal_mask, tgt_padding_mask):
            - src_padding_mask: Mask padding positions in source (1=real, 0=pad)
            - tgt_causal_mask: Lower triangular mask for causal attention
            - tgt_padding_mask: Mask padding positions in target

        Example:
            >>> src = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])  # 0 is pad
            >>> tgt = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
            >>> masks = model.create_masks(src, tgt)
            >>> src_padding_mask = masks[0]
            >>> src_padding_mask.shape
            torch.Size([2, 1, 1, 5])
        """
        batch_size, src_seq_len = src.shape
        tgt_seq_len = tgt.shape[1]
        # Convention: 1 = attend-to, 0 = mask-out (matches the causal mask).
        src_padding_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).to(src.device)
        tgt_padding_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_padding_mask, tgt_causal_mask, tgt_padding_mask

    def encode(self, src, src_padding_mask=None):
        """
        Encode source sequence through the encoder stack.

        The encoder processes the entire source sequence in parallel using
        self-attention. Each encoder layer applies:
        1. Multi-head self-attention (tokens attend to each other)
        2. Feed-forward network (transform each position independently)

        Args:
            src: Source token IDs of shape (batch, src_seq_len)
            src_padding_mask: Mask for padding positions, shape (batch, 1, 1, src_seq_len)
                              with 1=real token, 0=padding

        Returns:
            Encoder output of shape (batch, src_seq_len, d_model)

        Example:
            >>> model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000, d_model=64)
            >>> src = torch.randint(0, 1000, (2, 10))
            >>> src_padding_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            >>> encoder_output = model.encode(src, src_padding_mask)
            >>> encoder_output.shape
            torch.Size([2, 10, 64])
        """
        x = self.src_pos_encoding(self.src_embedding(src))
        for layer in self.encoder:
            x = layer(x, src_padding_mask)
        return x

    def decode(self, tgt, encoder_output, src_padding_mask, tgt_causal_mask, tgt_padding_mask):
        """
        Decode target sequence through the decoder stack.

        Args:
            tgt: Target token IDs of shape (batch, tgt_seq_len)
            encoder_output: Output from encoder of shape (batch, src_seq_len, d_model)
            src_padding_mask: Mask for source padding positions
            tgt_causal_mask: Lower triangular causal mask for self-attention
            tgt_padding_mask: Mask for target padding positions

        Returns:
            Decoder output of shape (batch, tgt_seq_len, d_model)

        Note: The actual masking happens inside MultiHeadAttention.forward() where
        mask=0 positions are filled with -1e9 before softmax, making their attention ~0.

        Example:
            >>> model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000, d_model=64)
            >>> tgt = torch.randint(0, 1000, (2, 8))
            >>> encoder_output = torch.randn(2, 10, 64)
            >>> src_padding_mask = torch.ones(2, 1, 1, 10)
            >>> tgt_causal_mask = torch.tril(torch.ones(8, 8))
            >>> decoder_output = model.decode(tgt, encoder_output, src_padding_mask, tgt_causal_mask, None)
            >>> decoder_output.shape
            torch.Size([2, 8, 64])
        """
        x = self.tgt_pos_encoding(self.tgt_embedding(tgt))
        # Combine causal + target padding so decoder self-attn also ignores pad positions.
        self_attn_mask = tgt_causal_mask
        if tgt_padding_mask is not None:
            self_attn_mask = tgt_causal_mask.bool() & tgt_padding_mask.bool()
        for layer in self.decoder:
            x = layer(x, encoder_output, self_attn_mask, src_padding_mask)
        return x

    def forward(self, src, tgt):
        """
        Forward pass of the Transformer model.

        Args:
            src: Source token IDs of shape (batch, src_seq_len)
            tgt: Target token IDs of shape (batch, tgt_seq_len)

        Returns:
            Logits tensor of shape (batch, tgt_seq_len, tgt_vocab_size)

        Example:
            >>> src = torch.randint(0, 1000, (2, 10))  # batch=2, src_len=10
            >>> tgt = torch.randint(0, 1000, (2, 8))   # batch=2, tgt_len=8
            >>> output = model(src, tgt)
            >>> output.shape
            torch.Size([2, 8, 1000])  # (batch, tgt_len, vocab_size)
        """
        src_pad, tgt_causal, tgt_pad = self.create_masks(src, tgt)
        encoder_output = self.encode(src, src_pad)
        decoder_output = self.decode(tgt, encoder_output, src_pad, tgt_causal, tgt_pad)
        return self.output_linear(decoder_output)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_lr_scheduler(base_lr: float, num_warmup_steps: int, current_step: int, total_steps: int):
    """
    Learning rate scheduler with warmup + cosine decay.

    The scheduler has two phases:
    1. Warmup phase: Linearly increase LR from 0 to base_lr
    2. Cosine decay phase: Gradually decrease LR following a cosine curve

    This schedule helps stabilize early training (warmup) and then
    fine-tune weights with smaller steps (decay).

    Args:
        base_lr: Base learning rate (peak value)
        num_warmup_steps: Number of warmup steps
        current_step: Current training step
        total_steps: Total training steps

    Returns:
        Learning rate for the current step

    Example:
        >>> # Warmup phase (increasing LR)
        >>> get_lr_scheduler(0.001, 100, 0, 1000)
        1e-05
        >>> get_lr_scheduler(0.001, 100, 50, 1000)
        0.0005
        >>> get_lr_scheduler(0.001, 100, 100, 1000)  # Peak
        0.001
        >>> # Cosine decay phase (decreasing LR)
        >>> get_lr_scheduler(0.001, 100, 550, 1000)  # Mid decay
        0.0007...
        >>> get_lr_scheduler(0.001, 100, 1000, 1000)  # End
        ~0.0
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
    """
    Train the model for one complete epoch.

    An epoch is one complete pass through the entire training dataset.
    This function handles:
    1. Forward pass: Compute model predictions
    2. Loss computation: Compare predictions to targets
    3. Backward pass: Compute gradients
    4. Gradient clipping: Prevent exploding gradients
    5. Learning rate scheduling: Adjust LR per step
    6. Weight update: Apply gradients via optimizer

    Args:
        model: The Transformer model to train
        dataloader: DataLoader yielding (src_batch, tgt_batch)
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device to run on (cpu/cuda/mps)
        current_step: Global training step counter
        total_steps: Total training steps for the full training run
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
        epoch: Current epoch number (1-indexed)
        total_epochs: Total number of epochs

    Returns:
        Tuple of (avg_loss, accuracy, new_current_step)

    Example:
        >>> # Assume model, dataloader, optimizer, criterion are set up
        >>> current_step = 0
        >>> for epoch in range(1, 11):
        ...     train_loss, train_acc, current_step = train_epoch(
        ...         model, dataloader, optimizer, criterion, device,
        ...         current_step, total_steps=1000, warmup_steps=100,
        ...         base_lr=0.001, epoch=epoch, total_epochs=10
        ...     )
        ...     print(f"Epoch {epoch}: loss={train_loss:.4f}")
    """
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

        # ─────────────────────────────────────────────────────────────────────
        # TEACHER FORCING: How we train the decoder to generate sequences
        # ─────────────────────────────────────────────────────────────────────
        # 
        # The Transformer decoder is trained to predict the NEXT token in a sequence.
        # We use "teacher forcing" — feeding the ground truth as input, shifted by one.
        #
        # Example: Vietnamese sentence = "tôi vui" (I am happy)
        # ─────────────────────────────────────────────────────────────────────
        # 
        # After tokenization with special tokens:
        #   tgt_batch = [<bos>, tôi, vui, <eos>]  (length 4)
        #               │       │    │     │
        #               │       │    │     └─ End of sequence
        #               │       │    └─ Token 2: "vui"
        #               │       └─ Token 1: "tôi"  
        #               └─ Start of sequence
        #
        # TEACHER FORCING SPLIT:
        # ─────────────────────────────────────────────────────────────────────
        # 
        # tgt_input  = tgt_batch[:, :-1]   # Decoder INPUT (what decoder sees)
        # tgt_target = tgt_batch[:, 1:]    # Decoder TARGET (what decoder should predict)
        #
        # UNDERSTANDING THE SLICING [:, :-1] and [:, 1:]:
        # ─────────────────────────────────────────────────────────────────────
        # 
        # Python slicing syntax: tensor[:, start:end]
        #   - First dimension (before comma): ":" means ALL batches
        #   - Second dimension (after comma): which positions to select
        #
        # [:, :-1]  = All batches, positions from START to SECOND-TO-LAST
        # [:, 1:]   = All batches, positions from SECOND to END
        #
        # VISUAL EXAMPLE with a batch of 2 sentences:
        # ─────────────────────────────────────────────────────────────────────
        #
        # tgt_batch shape: (2, 5)  # batch_size=2, seq_len=5
        #
        #   Batch 0: [<bos>, "tôi", "vui", "lắm", <eos>]
        #   Batch 1: [<bos>, "hello", "world", "!", <eos>]
        #
        # Applying [:, :-1] — "exclude the last column":
        # ─────────────────────────────────────────────────────────────────────
        #
        #   Index:     0        1       2       3        4
        #            [<bos>,  "tôi",  "vui",  "lắm",  <eos>]   ← Original
        #              │        │       │       │       │
        #              └────────┴───────┴───────┴───────┘
        #                         KEEP (exclude last)
        #
        #   Result: [<bos>, "tôi", "vui", "lắm"]   # Shape: (2, 4)
        #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #           All positions EXCEPT the last one
        #
        # Applying [:, 1:] — "exclude the first column":
        # ─────────────────────────────────────────────────────────────────────
        #
        #   Index:     0        1       2       3        4
        #            [<bos>,  "tôi",  "vui",  "lắm",  <eos>]   ← Original
        #              │        │       │       │       │
        #              └────────┴───────┴───────┴───────┘
        #                         └───────┴───────┴───────┘
        #                              KEEP (exclude first)
        #
        #   Result: ["tôi", "vui", "lắm", <eos>]   # Shape: (2, 4)
        #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #           All positions EXCEPT the first one
        #
        # SIDE-BY-SIDE COMPARISON:
        # ─────────────────────────────────────────────────────────────────────
        #
        #   Original:  [<bos>,  "tôi",  "vui",  "lắm",  <eos>]
        #                         │       │       │       │
        #   tgt_input:  [<bos>,  "tôi",  "vui",  "lắm"]      # [:, :-1] drop last
        #                         │       │       │       │
        #   tgt_target:         ["tôi",  "vui",  "lắm",  <eos>]  # [:, 1:] drop first
        #
        #   Notice: Each position in tgt_input aligns with the NEXT position in tgt_target
        #
        # WHY THIS ALIGNMENT?
        # ─────────────────────────────────────────────────────────────────────
        #
        # The model learns to predict the NEXT token given all previous tokens:
        #
        #   Step 0: Given [<bos>],              predict "tôi"
        #   Step 1: Given [<bos>, "tôi"],       predict "vui"
        #   Step 2: Given [<bos>, "tôi", "vui"], predict "lắm"
        #   Step 3: Given [<bos>, "tôi", "vui", "lắm"], predict <eos>
        #
        #   Input to decoder:  [<bos>, "tôi", "vui", "lắm"]
        #   Target to predict: ["tôi", "vui", "lắm", <eos>]
        #                         ↑       ↑       ↑       ↑
        #                    (shifted left by 1 position)
        #
        # This is "teacher forcing" because during training we feed the CORRECT
        # previous tokens (ground truth), not the model's own predictions.
        #
        # Why this works:
        #   - At each position i, decoder sees tokens 0..i and predicts token i+1
        #   - During training: we feed TRUE tokens (teacher forcing)
        #   - During inference: we feed PREDICTED tokens (autoregressive)
        #
        # src_input: Source sentence (English) for encoder
        #   - Encoder processes src_input to create contextual representations
        #   - Decoder attends to these via cross-attention
        # ─────────────────────────────────────────────────────────────────────
        
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
    """
    Evaluate the model on validation/test data.

    Unlike training, evaluation:
    1. Uses torch.no_grad() to skip gradient computation
    2. Feeds full target sequence (no teacher forcing shift)
    3. Computes loss and accuracy metrics

    Args:
        model: The Transformer model to evaluate
        dataloader: DataLoader yielding (src_batch, tgt_batch)
        criterion: Loss function
        device: Device to run on
        epoch: Optional epoch number for progress display
        total_epochs: Optional total epochs for progress display

    Returns:
        Tuple of (avg_loss, accuracy)

    Example:
        >>> # Evaluate on validation set
        >>> val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        >>> print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        Validation Loss: 2.3456, Accuracy: 0.7890
        >>> # With epoch display
        >>> val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch=5, total_epochs=10)
    """
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
    """
    Generate a Vietnamese translation from English text using greedy decoding.

    Greedy decoding generates one token at a time by always picking the
    highest probability token at each step. This is simple and fast,
    though not necessarily optimal (beam search is an alternative).

    The decoding process:
    1. Encode the source sentence
    2. Start with <BOS> token
    3. Decode to get probability distribution
    4. Pick the most likely token
    5. Repeat until <EOS> or max length

    Args:
        model: Trained Transformer model
        src_text: English source text string
        en_vocab: English tokenizer/vocabulary
        vi_vocab: Vietnamese tokenizer/vocabulary
        max_len: Maximum output length
        device: Device to run on

    Returns:
        Generated Vietnamese translation string

    Example — Step-by-Step Matrix Computations:
    ─────────────────────────────────────────────────────────────────────
    
    Let's trace how the model translates: "i am happy" → "tôi vui"
    
    Model Configuration (Tiny):
        - d_model = 64 (embedding dimension)
        - n_heads = 4 (attention heads)
        - d_head = d_model / n_heads = 16
        - src_vocab_size = 1000 (English)
        - tgt_vocab_size = 1000 (Vietnamese)
        - max_seq_len = 50
    
    STEP 0: Encode Source Sentence
    ─────────────────────────────────────────────────────────────────────
    
    Input: "i am happy"
    After tokenization: [2, 15, 42, 88, 3]  # [<bos>, "i", "am", "happy", <eos>]
    
    src_tensor shape: (1, 5)  # (batch_size=1, seq_len=5)
    
    src_padding_mask shape: (1, 1, 1, 5)
        - Created by: (src_tensor != pad_id).unsqueeze(1).unsqueeze(2)
        - Value: [1, 1, 1, 1, 1]  (all positions are valid)
    
    Encoder processes src_tensor:
        - Embedding: (1, 5) → (1, 5, 64)  # Each token → 64-dim vector
        - Positional encoding added
        - Through N encoder layers (self-attention + FFN)
    
    encoder_output shape: (1, 5, 64)
        - 5 positions, each with 64-dimensional contextual representation
        - This is K and V for cross-attention in decoder
    
    STEP 1: First Decoding Iteration (Predict First Token)
    ─────────────────────────────────────────────────────────────────────
    
    tgt_ids = [2]  # Just <bos>
    tgt_tensor shape: (1, 1)  # (batch=1, seq_len=1)
    
    causal_mask shape: (1, 1)
        [[1.]]  # Single position can see itself
    
    Decoder forward pass:
        tgt_tensor:     (1, 1)
        encoder_output: (1, 5, 64)
        src_padding_mask: (1, 1, 1, 5)
        causal_mask:    (1, 1)
    
    Inside Decoder Layer:
    ─────────────────────────────────────────────────────────────────────
    
    a) Masked Self-Attention:
       - Q, K, V all from tgt_tensor (decoder input)
       - Input shape: (1, 1, 64)  # after embedding + pos encoding
       - Q, K, V shapes: (1, 4, 1, 16)  # (batch, heads, seq_len, d_head)
       - attention output: (1, 1, 64)
    
    b) Cross-Attention:
       - Q from decoder self-attention: (1, 4, 1, 16)
       - K, V from encoder_output: (1, 4, 5, 16)  # 5 source positions
       - attention scores: (1, 4, 1, 5)  # attend to 5 source tokens
       - attention output: (1, 1, 64)
    
    c) FFN:
       - Input: (1, 1, 64)
       - Expanded: (1, 1, 128)  # d_ff = 128
       - Output: (1, 1, 64)
    
    decoder_output shape: (1, 1, 64)
    
    Output projection:
       logits = model.output_linear(decoder_output[:, -1, :])
       decoder_output[:, -1, :]: (1, 64)  # take last position
       output_linear: (1, 64) → (1, 1000)  # vocab_size
    
    logits shape: (1, 1000)
    probs = softmax(logits): (1, 1000)  # probability over all tokens
    
    Suppose highest probability is token 45 ("tôi"):
       next_token = 45
    
    STEP 2: Second Decoding Iteration (Predict Second Token)
    ─────────────────────────────────────────────────────────────────────
    
    tgt_ids = [2, 45]  # [<bos>, "tôi"]
    tgt_tensor shape: (1, 2)
    
    causal_mask shape: (2, 2)
        [[1., 0.],   # Position 0 can see: [0]
         [1., 1.]]   # Position 1 can see: [0, 1]
    
    Decoder forward pass:
        tgt_tensor:     (1, 2)
        encoder_output: (1, 5, 64)  # Reused from encoding step
        src_padding_mask: (1, 1, 1, 5)  # Reused
        causal_mask:    (2, 2)
    
    Inside Decoder Layer:
    ─────────────────────────────────────────────────────────────────────
    
    a) Masked Self-Attention:
       - Input: (1, 2, 64)  # 2 target tokens embedded
       - Q, K, V: (1, 4, 2, 16)
       - attention scores: (1, 4, 2, 2) masked with causal
       - attention output: (1, 2, 64)
    
    b) Cross-Attention:
       - Q from decoder: (1, 4, 2, 16)
       - K, V from encoder: (1, 4, 5, 16)
       - attention scores: (1, 4, 2, 5)  # 2 target positions × 5 source
       - Each target position attends to relevant source positions
       - attention output: (1, 2, 64)
    
    c) FFN:
       - Input: (1, 2, 64)
       - Output: (1, 2, 64)
    
    decoder_output shape: (1, 2, 64)
    
    Output projection (only last position):
       decoder_output[:, -1, :]: (1, 64)  # position 1 ("tôi")
       logits: (1, 1000)
       probs: (1, 1000)
    
    Suppose highest probability is token 72 ("vui"):
       next_token = 72
    
    STEP 3: Third Decoding Iteration (Predict <eos>)
    ─────────────────────────────────────────────────────────────────────
    
    tgt_ids = [2, 45, 72]  # [<bos>, "tôi", "vui"]
    tgt_tensor shape: (1, 3)
    
    causal_mask shape: (3, 3)
        [[1., 0., 0.],   # Position 0 can see: [0]
         [1., 1., 0.],   # Position 1 can see: [0, 1]
         [1., 1., 1.]]   # Position 2 can see: [0, 1, 2]
    
    Decoder forward pass (same pattern):
        decoder_output shape: (1, 3, 64)
        logits: (1, 1000)
    
    Suppose highest probability is token 3 (<eos>):
       next_token = 3  # Stop decoding!
    
    Generated: [45, 72]  # "tôi vui"
    
    KEY INSIGHTS:
    ─────────────────────────────────────────────────────────────────────
    
    1. Encoder runs ONCE, decoder runs for each generated token
    
    2. Cross-attention shapes:
       - Q: (batch, heads, tgt_len, d_head)
       - K, V: (batch, heads, src_len, d_head)
       - Output: (batch, heads, tgt_len, d_head)
    
    3. Causal mask grows each step:
       - Step 1: (1, 1)
       - Step 2: (2, 2)
       - Step 3: (3, 3)
    
    4. We only use the LAST position's output for prediction:
       decoder_output[:, -1, :] extracts the newest token's representation
    
    5. Matrix dimensions flow:
       (1, seq_len) → embed → (1, seq_len, 64) → attention → (1, seq_len, 64)
       → FFN → (1, seq_len, 64) → output_linear → (1, seq_len, 1000)
    """
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

    # Loss and optimizer.
    # Label smoothing = 0.1 (Vaswani et al. 2017): instead of pushing the
    # softmax toward a one-hot target, we push it toward "0.9 on the correct
    # token, 0.1 spread over the rest". Two effects:
    #   1. Raw cross-entropy can't reach zero, so the gradient keeps teaching
    #      the model to distribute mass sensibly across plausible alternatives
    #      (multiple valid translations) instead of over-committing to one.
    #   2. Slightly more stable optimization — loss is less peaky, gradients
    #      are less extreme on near-correct predictions.
    # Worth ~1–2 BLEU on IWSLT in practice.
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
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