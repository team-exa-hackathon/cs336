# Python Methods Cardio - Lecture 01 Essentials
## Stanford CS336: Language Modeling from Scratch

### 🎯 Essential Methods for Getting Started

---

## 📝 Text Processing Basics

```python
# Reading text data
with open('corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()                    # Read entire file as string
    
# Basic text cleaning
text = text.lower()                    # Lowercase everything
text = text.strip()                    # Remove leading/trailing whitespace
words = text.split()                   # Split into words (by whitespace)
lines = text.split('\n')               # Split into lines

# Character-level processing
chars = list(text)                     # Convert string to list of characters
unique_chars = sorted(set(text))       # Get unique characters
char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
```

---

## 🔢 Basic Tokenization

```python
# Simple word tokenization
def tokenize(text):
    return text.lower().split()

# Build vocabulary
def build_vocab(tokens):
    vocab = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

# Convert tokens to indices
def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]

# Example usage
tokens = tokenize("Hello world hello")
vocab = build_vocab(tokens)
indices = tokens_to_indices(tokens, vocab)
```

---

## 📊 Data Preparation

```python
# Create training sequences (for next-token prediction)
def create_sequences(text, seq_length):
    sequences = []
    for i in range(len(text) - seq_length):
        seq = text[i:i + seq_length]
        target = text[i + seq_length]
        sequences.append((seq, target))
    return sequences

# Batch data
def create_batches(data, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches
```

---

## 🐍 Python Essentials

```python
# List comprehensions (more Pythonic!)
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Dictionary comprehensions
word_counts = {word: text.count(word) for word in set(text.split())}

# Enumerate for index + value
for i, word in enumerate(words):
    print(f"Word {i}: {word}")

# Zip for parallel iteration
for token, index in zip(tokens, indices):
    print(f"{token} -> {index}")

# Collections Counter (useful for frequency counts)
from collections import Counter
word_freq = Counter(text.split())
most_common = word_freq.most_common(10)  # Top 10 words
```

---

## 🔥 PyTorch Basics

```python
import torch

# Creating tensors
x = torch.tensor([1, 2, 3])           # From list
x = torch.zeros(3, 4)                 # 3x4 zeros
x = torch.ones(2, 3)                  # 2x3 ones
x = torch.randn(3, 4)                 # Random normal

# Tensor operations
y = x + 1                             # Element-wise addition
z = x * 2                             # Element-wise multiplication
w = x @ y.T                           # Matrix multiplication

# Converting between numpy and torch
import numpy as np
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)   # Numpy to tensor
back_to_np = tensor.numpy()           # Tensor to numpy

# Basic neural network components
import torch.nn as nn
embedding = nn.Embedding(vocab_size, embedding_dim)
linear = nn.Linear(input_size, output_size)
```

---

## 📈 Simple Training Loop

```python
# Minimal training loop structure
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in data_loader:
        # Forward pass
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 💡 Quick Reference Card

| Task | Method | Example |
|------|--------|---------|
| Read file | `open()` + `.read()` | `text = open('file.txt').read()` |
| Split text | `.split()` | `words = text.split()` |
| Unique items | `set()` | `unique = set(words)` |
| Count items | `Counter()` | `Counter(words).most_common(5)` |
| Create mapping | dict comprehension | `{w: i for i, w in enumerate(words)}` |
| Batch data | list slicing | `data[i:i+batch_size]` |
| Create tensor | `torch.tensor()` | `torch.tensor([1, 2, 3])` |
| Train step | `.backward()` + `.step()` | See training loop above |

---

## 🚀 One-Liners for Common Tasks

```python
# Read entire file
text = open('data.txt', 'r').read()

# Get vocabulary size
vocab_size = len(set(text.split()))

# Create char-to-index mapping
c2i = {c: i for i, c in enumerate(sorted(set(text)))}

# Encode text to indices
encoded = [c2i[c] for c in text]

# Create bigrams
bigrams = [(text[i], text[i+1]) for i in range(len(text)-1)]

# Random sampling from list
import random
sample = random.choice(word_list)

# Save/load PyTorch model
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

---

*Pro tip: Start simple! Master these basics before moving to advanced methods.*