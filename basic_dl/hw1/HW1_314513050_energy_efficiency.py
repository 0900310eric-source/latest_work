import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Essential args
# batches = 30
epochs = 500
lr = 1e-4
FILE_PATH = '2025_energy_efficiency_data.csv'

# Architecture
class Heatingload:
  def __init__(self, lr, in_dim, hidden_dim1, hidden_dim2): # 15 -> 10 -> 10 -> 1
    self.lr = lr
    self.in_dim = in_dim
    self.hidden_dim1 = hidden_dim1
    self.hidden_dim2 = hidden_dim2

    # Layer
    np.random.seed(0)
    self.bias0 = np.zeros((1, hidden_dim1))
    self.W0 = np.random.randn(in_dim, hidden_dim1) * np.sqrt(1.0 / in_dim)
    self.bias1 = np.zeros((1, hidden_dim2))
    self.W1 = np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(1.0 / hidden_dim1)
    self.bias2 = np.zeros((1, 1))
    self.W2 = np.random.randn(hidden_dim2, 1) * np.sqrt(1.0 / hidden_dim2)
  def sigmoid(self, z):
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))
  def forward_propagation(self, x):
    self.x = x
    self.x0 = self.x @ self.W0 + self.bias0 # (N, 10)
    self.x1 = self.sigmoid(self.x0) # (N, 10)
    self.x2 = self.x1 @ self.W1 + self.bias1 # (N, 10)
    self.x3 = self.sigmoid(self.x2) # (N, 10)
    self.y_pred = self.x3 @ self.W2 + self.bias2 # (N, 1)
    return self.y_pred
  def backward_propagation(self, y):
    self.y = y.reshape(-1, 1)
    dE_dy_pred = 2 * (self.y_pred - self.y) # (N, 1)
    dsigmoid0 = self.x1 * (1 - self.x1) # (N, 10)
    dsigmoid2 = self.x3 * (1 - self.x3) # (N, 10)
    
    # output layer
    db2 = np.sum(dE_dy_pred, axis = 0, keepdims = True) # (1, 1)
    dW2 = self.x3.T @ dE_dy_pred # (10, 1)
    # hidden layer 2
    db1 = np.sum((dE_dy_pred @ self.W2.T) * dsigmoid2, axis = 0, keepdims = True) # (1, 10)
    dW1 = self.x1.T @ ((dE_dy_pred @ self.W2.T) * dsigmoid2) # (10, 10)
    # hidden layer 1
    db0 = np.sum((((dE_dy_pred @ self.W2.T) * dsigmoid2) @ self.W1.T) * dsigmoid0, axis = 0, keepdims = True)
    dW0 = self.x.T @ ((((dE_dy_pred @ self.W2.T) * dsigmoid2) @ self.W1.T) * dsigmoid0) 
    return db0, dW0, db1, dW1, db2, dW2
  def update(self, db0, dW0, db1, dW1, db2, dW2):
    self.bias0 -= self.lr * db0
    self.W0 -= self.lr * dW0
    self.bias1 -= self.lr * db1
    self.W1 -= self.lr * dW1
    self.bias2 -= self.lr * db2
    self.W2 -= self.lr * dW2

def rmse(y_hat, y): 
    y = y.reshape(-1,1)
    return np.sqrt(np.mean((y_hat - y)**2))

def train(model, x, y, mode):
  train_losses = []
  for epoch in range(epochs):
    model.forward_propagation(x)
    db0, dW0, db1, dW1, db2, dW2 = model.backward_propagation(y)
    model.update(db0, dW0, db1, dW1, db2, dW2)

    train_losses.append(rmse(model.y_pred, y))

    if (epoch + 1) % 10 == 0 and mode == 1:
      print(f"epoch {epoch+1:4d} | Train_RMSE {rmse(model.y_pred, y):.4f}")
    if (epoch + 1) % 10 == 0 and mode == 0:
      print(f"epoch {epoch+1:4d} | Test_RMSE {rmse(model.y_pred, y):.4f}")

  plt.figure()
  plt.plot(train_losses, label="Train RMSE")
  plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("Training curve")
  plt.grid(True); plt.legend(); plt.show()

def main():
  # Preprocessing
  df = pd.read_csv(FILE_PATH)
  labels = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution', 'Heating Load']
  df = df[labels] # Omitted cooling load
  one_hot_labels = ['Orientation', 'Glazing Area Distribution']
  df = pd.get_dummies(df, columns = one_hot_labels, drop_first = False, dtype = int)
  print(df.columns)
  drop_col = 'Glazing Area Distribution_0'
  df = df.drop(columns=[drop_col])
  
  df = df.rename(columns = {'Orientation_2': 'North', 'Orientation_3': 'South', 'Orientation_4': 'East', 'Orientation_5': 'West', 'Glazing Area Distribution_1': 'uniform', 'Glazing Area Distribution_2': 'north', 'Glazing Area Distribution_3': 'south', 'Glazing Area Distribution_4': 'east', 'Glazing Area Distribution_5': 'west'})
  
  # Shuffing dataframes
  df_shuffled = df.sample(frac = 1, random_state = 7).reset_index(drop = True) # replace old indexs

  # split data into training set and testing set (3:1)
  training_ratio = 0.75
  training_size = int(len(df_shuffled) * training_ratio)
  train_df_y = df_shuffled.loc[:training_size-1, 'Heating Load']
  test_df_y = df_shuffled.loc[training_size:, 'Heating Load']
  df_shuffled = df_shuffled.drop('Heating Load', axis = 1)
  train_df_x = df_shuffled.iloc[:training_size]
  test_df_x = df_shuffled.iloc[training_size:]

  cont = ['Relative Compactness','Surface Area','Wall Area','Roof Area',
        'Overall Height','Glazing Area']

  mu, sigma = train_df_x[cont].mean(), train_df_x[cont].std()
  train_df_x.loc[:, cont] = (train_df_x[cont] - mu) / sigma
  test_df_x.loc[:, cont]  = (test_df_x[cont]  - mu) / sigma

  # transform dataframes into numpy data
  train_y = train_df_y.to_numpy(np.float64)
  train_x = train_df_x.to_numpy(np.float64)
  test_y = test_df_y.to_numpy(np.float64)
  test_x = test_df_x.to_numpy(np.float64)

  # training
  heatingload = Heatingload(lr, 15, 10, 10)
  train(heatingload, train_x, train_y, 1)

  # test
  y_pred_train = heatingload.forward_propagation(train_x)
  train_rmse_final = rmse(y_pred_train, train_y)

  y_pred_test  = heatingload.forward_propagation(test_x)
  test_rmse_final  = rmse(y_pred_test,  test_y)

  print(f"Final Train RMSE: {train_rmse_final:.4f}")
  print(f"Final Test  RMSE: {test_rmse_final:.4f}")

  # regression result - TRAIN
  plt.figure()
  plt.plot(train_y, label="label", color="blue")
  plt.plot(y_pred_train.squeeze(), label="predict", color="red", alpha=0.7)
  plt.title("prediction for training data")
  plt.ylabel("heating load"); plt.xlabel("#th case")
  plt.legend(); plt.grid(True); plt.show()

  # regression result - TEST
  plt.figure()
  plt.plot(test_y, label="label", color="blue")
  plt.plot(y_pred_test.squeeze(), label="predict", color="red", alpha=0.7)
  plt.title("prediction for test data")
  plt.ylabel("heating load"); plt.xlabel("#th case")
  plt.legend(); plt.grid(True); plt.show()

  # # Covariance Matrix
  # cov_matrix = df.cov().to_numpy()

  # plt.imshow(cov_matrix, cmap="coolwarm", interpolation="nearest")
  # plt.colorbar(label="Covariance")
  # plt.title("Covariance Matrix")
  # plt.xticks(range(len(df.columns)), df.columns, rotation=90)
  # plt.yticks(range(len(df.columns)), df.columns)
  # plt.tight_layout()
  # plt.show()

  # features (15 dims) 和 Heating Load
  feature_cols = [c for c in df.columns if c != 'Heating Load']
  corr_with_y = df[feature_cols].corrwith(df['Heating Load'])

  print("Correlation with Heating Load:")
  print(corr_with_y.sort_values(ascending=False))

  # (Bar chart）
  plt.figure(figsize=(8,6))
  corr_with_y.sort_values().plot(kind="barh", color="skyblue")
  plt.title("Correlation of Features with Heating Load")
  plt.xlabel("Correlation coefficient")
  plt.grid(True, axis="x", linestyle="--", alpha=0.7)
  plt.show()

if __name__ == '__main__':
  main()