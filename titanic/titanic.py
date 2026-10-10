import numpy as np
import pandas as pd

# -----------------------------
# Load and prepare data
# -----------------------------
data = pd.read_csv('Titanic-Dataset.csv')

data = data.drop(columns=["Ticket", "Fare", "PassengerId", "Name", "Cabin", "Embarked"])
data = data.dropna(subset=["Age"])

data["Sex"] = data["Sex"].map({
    "male": 0,
    "female": 1
})

data = data.sample(frac=1, random_state=42)

split_index = int(0.8 * len(data))

train = data.iloc[:split_index]
test = data.iloc[split_index:]

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# -----------------------------
# Load train/test data
# -----------------------------
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

X_train = train_data[features].to_numpy(dtype=float)
y_train = train_data["Survived"].to_numpy(dtype=float)

X_test = test_data[features].to_numpy(dtype=float)
y_test = test_data["Survived"].to_numpy(dtype=float)

# -----------------------------
# Optional but recommended: feature scaling
# -----------------------------
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1.0

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# -----------------------------
# Regularized logistic regression
# -----------------------------
m, n = X_train.shape

epochs = 1000
alpha = 0.01
reg_lambda = 0.1   # L2 regularization strength; try 0.01, 0.1, 1.0

W = np.zeros(n)
b = 0.0

for epoch in range(epochs):
    # Forward pass
    z = X_train @ W + b
    f_x = 1 / (1 + np.exp(-z))

    # Cost with L2 regularization
    eps = 1e-15
    loss = -y_train * np.log(f_x + eps) - (1 - y_train) * np.log(1 - f_x + eps)

    cost = np.mean(loss) + (reg_lambda / (2 * m)) * np.sum(W ** 2)

    # Gradients
    error = f_x - y_train

    dj_dw = (X_train.T @ error) / m + (reg_lambda / m) * W
    dj_db = np.sum(error) / m

    # Update parameters
    W -= alpha * dj_dw
    b -= alpha * dj_db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | cost = {cost:.6f}")

# -----------------------------
# Evaluation
# -----------------------------
z_test = X_test @ W + b
probs = 1 / (1 + np.exp(-z_test))
preds = (probs >= 0.5).astype(int)

accuracy = np.mean(preds == y_test) * 100
print("Test accuracy:", accuracy)