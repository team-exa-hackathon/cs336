# Python Methods Cardio - Language Modeling Lecture 01
## Stanford CS336 Spring 2025

This is a quick reference guide for Python methods commonly used in language modeling.

---

## 🐍 Core Python Methods

### String Methods
```python
# Text processing essentials
text.lower()              # Convert to lowercase
text.upper()              # Convert to uppercase
text.strip()              # Remove whitespace from ends
text.split()              # Split string into list (default: by whitespace)
text.split('\n')          # Split by newlines
text.replace('old', 'new') # Replace substring
text.startswith('prefix') # Check if starts with prefix
text.endswith('suffix')   # Check if ends with suffix
text.join(list)           # Join list elements with string as separator
' '.join(['hello', 'world']) # → 'hello world'
text.encode('utf-8')      # Encode string to bytes
bytes.decode('utf-8')     # Decode bytes to string
```

### List Methods
```python
# List manipulation
lst.append(item)          # Add item to end
lst.extend(iterable)      # Add all items from iterable
lst.insert(idx, item)     # Insert item at index
lst.remove(item)          # Remove first occurrence
lst.pop(idx=-1)           # Remove and return item at index
lst.index(item)           # Find index of first occurrence
lst.count(item)           # Count occurrences
lst.sort()                # Sort in place
lst.reverse()             # Reverse in place
lst.clear()               # Remove all items
```

### Dictionary Methods
```python
# Dictionary operations
dict.get(key, default)    # Get value with default
dict.keys()               # Get all keys
dict.values()             # Get all values
dict.items()              # Get (key, value) pairs
dict.update(other_dict)   # Update with another dict
dict.pop(key, default)    # Remove and return value
dict.setdefault(key, val) # Set if key doesn't exist
dict.clear()              # Remove all items
```

### File I/O Methods
```python
# File operations
with open('file.txt', 'r') as f:
    content = f.read()           # Read entire file
    lines = f.readlines()        # Read all lines into list
    line = f.readline()          # Read single line

with open('file.txt', 'w') as f:
    f.write('text')              # Write string
    f.writelines(lines)          # Write list of strings

# Path operations
import os
os.path.exists(path)             # Check if path exists
os.path.join(dir, file)          # Join path components
os.listdir(directory)            # List directory contents
os.makedirs(path, exist_ok=True) # Create directories
```

---

## 🔢 NumPy Methods

### Array Creation & Manipulation
```python
import numpy as np

# Creation
np.array([1, 2, 3])              # Create array
np.zeros((3, 4))                 # Create zeros array
np.ones((3, 4))                  # Create ones array
np.arange(0, 10, 2)              # Create range array
np.linspace(0, 1, 100)           # Create evenly spaced values
np.random.randn(3, 4)            # Random normal distribution
np.random.randint(0, 10, (3, 4)) # Random integers

# Shape manipulation
arr.reshape(new_shape)           # Reshape array
arr.transpose()                  # Transpose array
arr.T                            # Transpose (shorthand)
arr.flatten()                    # Flatten to 1D
np.concatenate([a1, a2], axis=0) # Concatenate arrays
np.stack([a1, a2], axis=0)       # Stack arrays
np.split(arr, n, axis=0)         # Split array
```

### Mathematical Operations
```python
# Element-wise operations
np.sum(arr, axis=0)              # Sum along axis
np.mean(arr, axis=0)             # Mean along axis
np.std(arr, axis=0)              # Standard deviation
np.max(arr, axis=0)              # Maximum along axis
np.min(arr, axis=0)              # Minimum along axis
np.argmax(arr, axis=0)           # Index of maximum
np.argmin(arr, axis=0)           # Index of minimum

# Linear algebra
np.dot(a, b)                     # Dot product
a @ b                            # Matrix multiplication
np.matmul(a, b)                  # Matrix multiplication
np.linalg.norm(arr)              # Norm of array
np.linalg.inv(matrix)            # Matrix inverse
```

---

## 🔥 PyTorch Methods

### Tensor Operations
```python
import torch

# Creation
torch.tensor([1, 2, 3])          # Create tensor
torch.zeros((3, 4))              # Create zeros tensor
torch.ones((3, 4))               # Create ones tensor
torch.randn(3, 4)                # Random normal
torch.randint(0, 10, (3, 4))     # Random integers
torch.arange(0, 10, 2)           # Range tensor
torch.linspace(0, 1, 100)        # Evenly spaced

# Shape operations
tensor.view(new_shape)           # Reshape tensor
tensor.reshape(new_shape)        # Reshape (alternative)
tensor.transpose(0, 1)           # Transpose dimensions
tensor.permute(2, 0, 1)          # Permute dimensions
tensor.squeeze()                 # Remove size-1 dimensions
tensor.unsqueeze(dim)            # Add size-1 dimension
torch.cat([t1, t2], dim=0)       # Concatenate
torch.stack([t1, t2], dim=0)     # Stack tensors
```

### Neural Network Methods
```python
import torch.nn as nn
import torch.nn.functional as F

# Layer creation
nn.Linear(in_features, out_features)     # Linear layer
nn.Embedding(num_embeddings, embedding_dim) # Embedding layer
nn.LayerNorm(normalized_shape)            # Layer normalization
nn.Dropout(p=0.1)                         # Dropout layer

# Activation functions
F.relu(x)                                 # ReLU
F.gelu(x)                                 # GELU
F.softmax(x, dim=-1)                      # Softmax
F.log_softmax(x, dim=-1)                  # Log softmax
torch.sigmoid(x)                          # Sigmoid
torch.tanh(x)                             # Tanh

# Loss functions
F.cross_entropy(logits, targets)          # Cross entropy loss
F.mse_loss(pred, target)                  # MSE loss
F.nll_loss(log_probs, targets)            # Negative log likelihood
```

### Training Methods
```python
# Optimizer methods
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()            # Clear gradients
optimizer.step()                 # Update parameters

# Model methods
model.train()                    # Set to training mode
model.eval()                     # Set to evaluation mode
model.parameters()               # Get model parameters
model.state_dict()               # Get state dictionary
model.load_state_dict(state)     # Load state dictionary

# Gradient methods
tensor.backward()                # Compute gradients
tensor.detach()                  # Detach from computation graph
tensor.requires_grad_(True)      # Enable gradient computation
with torch.no_grad():            # Disable gradient computation
    # inference code
```

---

## 📊 Data Processing Methods

### Tokenization & Text Processing
```python
# Basic tokenization
text.split()                     # Whitespace tokenization
re.split(r'\W+', text)           # Split on non-word chars
re.findall(r'\w+', text)         # Find all words

# Common preprocessing
import re
re.sub(r'[^\w\s]', '', text)    # Remove punctuation
re.sub(r'\s+', ' ', text)        # Normalize whitespace
text.lower().strip()             # Lowercase and strip

# Encoding/Decoding
vocab = {word: idx for idx, word in enumerate(words)}
inverse_vocab = {idx: word for word, idx in vocab.items()}

def encode(text, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in text.split()]

def decode(indices, inverse_vocab):
    return ' '.join([inverse_vocab.get(idx, '<UNK>') for idx in indices])
```

### Batch Processing
```python
# Creating batches
def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Padding sequences
def pad_sequences(sequences, max_len, pad_value=0):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [pad_value] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return padded

# Collate function for DataLoader
def collate_fn(batch):
    texts, labels = zip(*batch)
    # Process and pad
    return torch.stack(texts), torch.tensor(labels)
```

---

## 🔧 Utility Methods

### Random & Reproducibility
```python
# Set random seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### Progress Tracking
```python
# Using tqdm for progress bars
from tqdm import tqdm

for i in tqdm(range(100), desc="Training"):
    # training code
    pass

# Manual progress tracking
for i, batch in enumerate(dataloader):
    if i % 100 == 0:
        print(f"Step {i}/{len(dataloader)}")
```

### Memory Management
```python
# PyTorch memory management
torch.cuda.empty_cache()         # Clear GPU cache
torch.cuda.memory_allocated()    # Check allocated memory
torch.cuda.memory_reserved()     # Check reserved memory

# Delete references
del tensor
import gc
gc.collect()                     # Force garbage collection
```

---

## 📝 Common Patterns

### Context Managers
```python
# File handling
with open('file.txt', 'r') as f:
    content = f.read()

# Gradient context
with torch.no_grad():
    outputs = model(inputs)

# Device context
with torch.cuda.device(0):
    tensor = torch.randn(3, 4).cuda()
```

### List Comprehensions
```python
# Basic comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]

# Nested comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]

# Dictionary comprehension
word_lengths = {word: len(word) for word in text.split()}
```

### Lambda Functions
```python
# Sorting with key
words.sort(key=lambda x: len(x))

# Mapping
lengths = list(map(lambda x: len(x), words))

# Filtering
long_words = list(filter(lambda x: len(x) > 5, words))
```

---

## 🎯 Quick Tips

1. **Use vectorized operations** instead of loops when possible
2. **Check tensor shapes** frequently with `.shape` or `.size()`
3. **Move tensors to GPU** with `.cuda()` or `.to(device)`
4. **Use `torch.no_grad()` during inference** to save memory
5. **Set random seeds** for reproducibility
6. **Use descriptive variable names** for clarity
7. **Profile your code** to find bottlenecks
8. **Save checkpoints regularly** during training

---

*Remember: This is a reference guide. Always check the official documentation for the most up-to-date information and additional parameters.*