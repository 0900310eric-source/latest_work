import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = '2025_ionosphere_data.csv'
lr = 1e-3
epochs = 500
np.random.seed(42) # Random seed

class Ionosphere:
  def __init__(self, lr, in_dim, hid1_dim, hid2_dim, out_dim):
    self.lr = lr
    self.in_dim = in_dim
    self.hid1_dim = hid1_dim
    self.hid2_dim = hid2_dim
    self.out_dim = out_dim

    #layer
    self.b = []
    self.W = []
    b0 = np.zeros((1, self.hid1_dim))
    W0 = np.random.randn(in_dim, hid1_dim) * np.sqrt(1 / self.in_dim)
    b1 = np.zeros((1, self.hid2_dim))
    W1 = np.random.randn(self.hid1_dim, self.hid2_dim) * np.sqrt(1 / self.hid1_dim)
    b2 = np.zeros((1, self.out_dim))
    W2 = np.random.randn(self.hid2_dim, self.out_dim) * np.sqrt(1 / self.hid2_dim)
    self.b.extend([b0, b1, b2])
    self.W.extend([W0, W1, W2])

  # Activate functions
  def Relu(self, z):
    return np.maximum(0, z)
  def softmax(self, z):
    z = np.clip(z, -50, 50)
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)
  def cross_entropy(self, y_pred, y):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0)
    return -np.sum(y * np.log(y_pred))

  def relu_grad(self, z):
    return (z > 0).astype(float)
  def forward(self, x):
    self.x = x
    self.x0 = self.x @ self.W[0] + self.b[0]
    self.x1 = self.Relu(self.x0)
    self.x2 = self.x1 @ self.W[1] + self.b[1]
    self.x3 = self.Relu(self.x2)
    self.x4 = self.x3 @ self.W[2] + self.b[2]
    self.y_pred = self.softmax(self.x4)
    return self.y_pred
  def backward(self, y):
    db = []
    dW = []
    self.y = y
    
    # N = y.shape[0]
    # dE_dypred = -self.softmax(y) / self.y_pred # (N, 2)
    # d_softmax = self.y_pred * reversed(self.y_pred) # (N, 2)
    delta = (self.y_pred - y)
    db2 = np.sum(delta, axis = 0, keepdims = True) # (1, 2)
    dW2 = self.x3.T @ (delta) # (17, 2)
    db1 = np.sum(((delta) @ self.W[2].T) * self.relu_grad(self.x2), axis = 0, keepdims = True) # (1, 17)
    dW1 = self.x1.T @ (((delta) @ self.W[2].T) * self.relu_grad(self.x2)) # (17, 17)
    db0 = np.sum( ((delta @ self.W[2].T) * self.relu_grad(self.x2)) @ self.W[1].T * self.relu_grad(self.x0), axis=0, keepdims=True )# (1, 17)
    dW0 = self.x.T @ ( ((delta @ self.W[2].T) * self.relu_grad(self.x2)) @ self.W[1].T * self.relu_grad(self.x0) ) # (34, 17)

    db.extend([db0, db1, db2])
    dW.extend([dW0, dW1, dW2])
    return db, dW
  def update(self, db, dW):
    for i in range(len(self.b)):
      self.b[i] -= self.lr * db[i]
      self.W[i] -= self.lr * dW[i]
  def accuracy(self, y_pred, y):
    pred_class = np.argmax(y_pred, axis=1)
    true_class = np.argmax(y, axis=1)
    return np.mean(pred_class == true_class)
  def train(self, x, y):
    train_losses = []
    capture_epochs = [10, 390]  # 
    captured = {}
    for epoch in range(epochs):
      self.forward(x)
      db, dW = self.backward(y)
      self.update(db, dW)

      # loss and accuracy
      train_loss = self.cross_entropy(self.y_pred, y)
      acc = self.accuracy(self.y_pred, y)
      train_losses.append(train_loss)

      if (epoch + 1) in capture_epochs:
        feats = self.x3.copy()
        if feats.shape[1] != 2:
          feats = pca_2d(feats)  # using numpy -> 2D
        labels = np.argmax(y, axis=1)
        captured[epoch + 1] = (feats, labels)

      if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1:4d} | Train_Cross_Entropy {train_losses[epoch]:.4f} | acc {acc * 100:.4f} %")

    plt.figure()
    plt.plot(train_losses, label="Train Cross Entropy")
    plt.xlabel("Epoch"); plt.ylabel("Cross Entropy"); plt.title("Training curve")
    plt.grid(True); plt.legend(); plt.show()
    for ep in capture_epochs:
      feats, labels = captured[ep]
      plt.figure(figsize=(4, 4))
      plt.scatter(feats[labels == 0, 0], feats[labels == 0, 1], color='blue', label='Class 1')
      plt.scatter(feats[labels == 1, 0], feats[labels == 1, 1], color='red', label='Class 2')
      plt.title(f"2D features before output layer (epoch {ep})")
      plt.xlabel("feature 1")
      plt.ylabel("feature 2")
      plt.legend()
      plt.tight_layout()
      plt.show()

def pca_2d(X):
  X = X - np.mean(X, axis=0, keepdims=True)
  U, S, Vt = np.linalg.svd(X, full_matrices=False)
  return X @ Vt[:2].T

def main():
  # Read the file
  df = pd.read_csv(FILE_PATH, header = None)
  # print(df.head(1))

  # Shuffle the dataframe
  df_shuffled = df.sample(frac = 1, random_state = 42).reset_index(drop = True) # shuffle the samples and reset the indexs
  # print(df_shuffled.head(1))
  train_ratio = 0.8 # train: 80%, test: 20%
  train_len = int(train_ratio * df_shuffled.shape[0])
  df_train = df_shuffled.iloc[:train_len, :]
  df_test = df_shuffled.iloc[train_len:, :]

  # split data into train set and test set
  df_train_x = df_train.iloc[:, :-1]
  df_train_y = df_train.iloc[:, -1]
  df_train_y = df_train_y.map({'g': 1, 'b': 0})
  df_test_x = df_test.iloc[:, :-1]
  df_test_y = df_test.iloc[:, -1]
  df_test_y = df_test_y.map({'g': 1, 'b': 0})
  # print(df_train_y)

  # Transform pd data into numpy data
  train_x = df_train_x.to_numpy()
  train_y = df_train_y.to_numpy()
  train_y = np.eye(2)[train_y]
  print(train_y.shape, train_x.shape)
  test_x = df_test_x.to_numpy()
  test_y = df_test_y.to_numpy()
  test_y = np.eye(2)[test_y]

  # Training
  ionosphere = Ionosphere(lr, 34, 17, 17, 2)
  ionosphere.train(train_x, train_y)

  # Testing
  y_test_pred = ionosphere.forward(test_x)
  test_acc = ionosphere.accuracy(y_test_pred, test_y)
  print(f"Final test accuracy: {test_acc * 100:.4f} %")

if __name__ == '__main__':
  main()