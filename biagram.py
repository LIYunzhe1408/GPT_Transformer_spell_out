import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel
block_size = 8  # what is time maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2

# Data when loading, model(parameters) when constructing should be moved to device
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# -------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# All the unique characters that occur in this text
characters = sorted(list(set(text)))
vocab_size = len(characters)
# Create a mapping from characters that occur in this text
itos = {i: ch for i, ch in enumerate(characters)}
stoi = {ch: i for i, ch in itos.items()}
encode = lambda string: [stoi[ch] for ch in string] # Encoder: take a string, output a list of integers
decode = lambda integers: ''.join([itos[i] for i in integers]) # Decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]


# Data loading
def get_batch(split):
  data = train_data if split == 'train' else val_data
  index_x = torch.randint(len(data) - block_size, (batch_size,))
  xb = torch.stack([data[i:i+block_size] for i in index_x])
  yb = torch.stack([data[i+1:i+block_size+1] for i in index_x])
  xb, yb = xb.to(device), yb.to(device)
  return xb, yb

# Use this to indicate to PyTorch we will not call loss.backward() at all
# As it will not store every intermediate variables -> ++ efficient
@torch.no_grad() 
def estimate_loss():
  "Some layers will have different behavior in training and inference time"
  out = {}
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out



# Model
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    # Each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    # idx and targets are both (B, T) tensors of integers
    logits = self.token_embedding_table(idx) # (B, T, C)
    
    # Just inference
    if targets is None:
      return logits, None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
      return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx: (B, T)
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      # For Biagram model, we just see the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # Apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # Sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # Append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
  
    if step % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {step}: train loss {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch("train")

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
idx = model.generate(context, max_new_tokens=500)[0]
print(decode(idx.tolist()))