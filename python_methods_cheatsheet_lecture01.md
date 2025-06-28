# Python Methods Cheat Sheet - Language Modeling
### CS336 Quick Reference

## Text Processing
```python
text = open('file.txt', 'r').read()     # Read file
text.lower().strip().split()            # Clean & tokenize
set(text.split())                       # Unique words
''.join(words)                          # Join words
```

## Data Structures
```python
# Lists
lst.append(x)         # Add to end
lst.extend([x,y])     # Add multiple
lst[i:j]              # Slice
[x**2 for x in lst]   # List comp

# Dicts
d = {k: v for k, v in pairs}  # Dict comp
d.get(key, default)           # Safe access
d.keys(), d.values()          # Get keys/vals
Counter(lst).most_common(n)   # Frequency
```

## NumPy
```python
np.array([1,2,3])            # Create array
np.zeros((m,n))              # Zero matrix
arr.shape                    # Dimensions
arr.reshape(m,n)             # Reshape
arr @ other                  # Matrix mult
np.mean(arr, axis=0)         # Mean along axis
```

## PyTorch Tensors
```python
torch.tensor([1,2,3])        # Create tensor
torch.zeros(m,n)             # Zero tensor
x.view(-1, n)                # Reshape
x.transpose(0,1)             # Transpose
x @ y                        # Matrix mult
x.to('cuda')                 # Move to GPU
```

## PyTorch NN
```python
# Layers
nn.Embedding(vocab_size, dim)
nn.Linear(in_dim, out_dim)
nn.LayerNorm(dim)
nn.Dropout(p=0.1)

# Activations
F.relu(x)
F.softmax(x, dim=-1)
torch.sigmoid(x)

# Loss
F.cross_entropy(logits, targets)
```

## Training Pattern
```python
# Setup
model = Model()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train step
outputs = model(inputs)       # Forward
loss = criterion(outputs, targets)
optim.zero_grad()            # Clear grads
loss.backward()              # Backward
optim.step()                 # Update

# Inference
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

## Tokenization
```python
# Build vocab
vocab = {word: i for i, word in enumerate(words)}
inverse = {i: word for word, i in vocab.items()}

# Encode/decode
encoded = [vocab[w] for w in text.split()]
decoded = ' '.join([inverse[i] for i in encoded])
```

## Useful Patterns
```python
# Batch processing
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]

# Enumerate
for i, item in enumerate(items):
    print(f"{i}: {item}")

# Zip
for x, y in zip(list1, list2):
    process(x, y)

# Context manager
with open('file.txt') as f:
    content = f.read()
```

## Quick Commands
```python
len(x)                       # Length
sorted(x)                    # Sort
set(x)                       # Unique
list(set(x))                 # Unique list
x[::-1]                      # Reverse
random.choice(lst)           # Random pick
' '.join(map(str, lst))      # List to string
```

---
*Remember: `dir(obj)` shows available methods, `help(method)` shows docs*