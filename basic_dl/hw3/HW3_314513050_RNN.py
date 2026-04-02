import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import io

Files = ['shakespeare_train.txt', 'shakespeare_valid.txt']
lr = 5e-4
BATCH_SIZE = 64
epochs = 10

class SimpleRNN(nn.Module):
  def __init__(self, vocab_size: int, hidden_size: int):
    super().__init__()
    self.V = vocab_size # 67
    self.H = hidden_size # 256

    # Parameters
    self.W_xh = nn.Parameter(torch.randn(self.V, self.H) / (self.V ** 0.5))
    self.W_hh = nn.Parameter(torch.randn(self.H, self.H) / (self.H ** 0.5))
    self.b_h  = nn.Parameter(torch.zeros(self.H))
    self.W_hy = nn.Parameter(torch.randn(self.H, self.V) / (self.H ** 0.5))
    self.b_y  = nn.Parameter(torch.zeros(self.V))

  def forward(self, X_one_hot, h0 = None):
    # """
    #     X_one_hot: (B, T, V)  one-hot batch
    #     h0  : (B, H)     initial hidden state；
    #     If None → 0, then return:
    #       logits: (B, T, V)
    #       h_T   : (B, H)
    #     """
    B, T, V = X_one_hot.shape
    H = self.H
    if h0 is None:
      h_t = X_one_hot.new_zeros(B, H)
    else:
      h_t = h0

    logits_list = []
    # Time_seqs circuits
    for t in range(T):
      x_t = X_one_hot[:, t, :]          # (B, V)
      # h_t = tanh(x_t*W_xh + h_{t-1}*W_hh + b_h)
      h_t = torch.tanh(x_t @ self.W_xh + h_t @ self.W_hh + self.b_h)
      # y_t = h_t*W_hy + b_y  → logits
      y_t = h_t @ self.W_hy + self.b_y
      logits_list.append(y_t.unsqueeze(1))   # (B,1,V)

    logits = torch.cat(logits_list, dim=1)     # (B,T,V)
    return logits, h_t

class CharSeqDataset(Dataset):
  def __init__(self, x_np, y_np):
    self.X = torch.from_numpy(x_np.astype(np.int64)) # (_, 100)
    self.Y = torch.from_numpy(y_np.astype(np.int64)) # (_, 100)
  def __len__(self):
    return self.X.shape[0]
  def __getitem__(self, index):
    return self.X[index], self.Y[index] # (100, )

def txt_to_numpy(files, index, vocab_to_int = None):
  with io.open(files[index], 'r', encoding = 'utf-8') as f:
    text = f.read() # Read the whole file

  # Construct two mappings
  if vocab_to_int is None:
    vocabs = sorted(set(text)) # The collection of vocabularies. Here, I used sorted() to make sure the validation set would follow the same vocabs (order).
    vocab_to_int = {c: i for i, c  in enumerate(vocabs)} # train: 67
    int_to_vocab = {i: c for i, c in enumerate(vocabs)} # train: 67
  else:
    int_to_vocab = {i: c for i, c in vocab_to_int.items()}

  # Encode (list to numpy)
  data = np.array([vocab_to_int[vocab] for vocab in text], dtype = np.int32) # train: 4351312
  return data, vocab_to_int, int_to_vocab

def one_hot_encoding(data, encoding_size):
  return np.eye(encoding_size)[data]

def numpy_shuffling(data, train_size, seq_size):

  # To generate a list of the shuffled indexs.
  shuffle_size = (train_size // seq_size) * seq_size
  shuffle_index_list = np.arange(0, shuffle_size, seq_size)
  np.random.shuffle(shuffle_index_list)
  
  # To preserve some issues with unknown reasons
  copied_data = data.copy()
  # Shuffling
  for i, v in enumerate(shuffle_index_list):
    key = i * seq_size
    data[key:key+seq_size] = copied_data[v:v+seq_size] # Omit the problem of tail, because the tail in this scenario is too small.
  return

def numpy_seq_pairs(data, train_size, seq_size):

  # Remove the tail
  shifting_size = (train_size // seq_size) * seq_size
  data_shifted = data[:shifting_size]

  # Reshape
  data_shifted = data_shifted.reshape((-1, seq_size))
  # print(data_shifted.shape) # (71332, 101)

  # ***** Send x, y to dataset *****
  x = data_shifted[:, :seq_size-1]
  y = data_shifted[:, 1:]
  print(x.shape[1])
  return x, y

def main():
  # 1. Preprocessing

  # Convert characters to numbers (training set), training
  train_data, vocab_to_int, int_to_vocab = txt_to_numpy(Files, 0)
  # print(train_data[0:100])

  # Validation
  valid_data, _, _ = txt_to_numpy(Files, 1, vocab_to_int)

  # data shuffling
  sequence_size = 101
  train_data_size = train_data.shape[0]

  numpy_shuffling(train_data, train_data_size, sequence_size)

  # 2. Dataset and Dataloader
  # In order to maintain the size of seq_size
  train_x_np, train_y_np = numpy_seq_pairs(train_data, train_data_size, sequence_size) 
  train_ds = CharSeqDataset(train_x_np, train_y_np)
  train_dataloader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = False)

  # For check
  # X_batch, Y_batch = next(iter(train_dataloader))
  # print(X_batch.shape, Y_batch.shape)
  # assert torch.all((X_batch[:, 1:] == Y_batch[:, :-1])), "Wrong"
  # diff = (X_batch[:, 1:] != Y_batch[:, :-1])
  # print("aligned:", (~diff).all().item(), "mismatches:", diff.sum().item())

  # 3. Training
  V = len(vocab_to_int)
  H = 256 # Hidden size

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  simpleRNN = SimpleRNN(V, H).to(device)
  opt = optim.Adam(simpleRNN.parameters(), lr = lr)
  cross_entropies = nn.CrossEntropyLoss()
  
  simpleRNN.train()
  for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_error = 0.0
    for X_batch, Y_batch in train_dataloader:
      X_batch = X_batch.to(device)
      Y_batch = Y_batch.to(device)
      # One hot encoding
      X_batch_onehot = F.one_hot(X_batch, num_classes = V).float().to(device) # torch.Size([64, 100, 67])
      logits, _ = simpleRNN(X_batch_onehot)                             # (B,100,V)
      loss = cross_entropies(logits.reshape(-1, V), Y_batch.reshape(-1))
      opt.zero_grad(); loss.backward(); opt.step()

      with torch.no_grad():
        pred = logits.argmax(dim = -1)
        train_error = (1.0 - (pred == Y_batch).float().mean().item()) * float(100)

    # Calculate epoch loss and epoch error
      epoch_loss += loss.item()
      epoch_error += train_error
    
    epoch_loss /= len(train_dataloader)
    epoch_error /= len(train_dataloader)
    print(f"epoch {epoch+1}: loss={epoch_loss:.4f}, train_err={epoch_error:.4f}")


if __name__ == '__main__':
  main()