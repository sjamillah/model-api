# models/gradient_descent.py
import numpy as np
import pandas as pd

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000, batch_size=32, print_cost=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.training_loss = []
        self.validation_loss = []
        self.print_cost = print_cost

    def fit(self, X, y, validation_split=0.1):
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros((n_features, y.shape[1]))
        self.bias = np.zeros(y.shape[1])

        # Split validation set
        n_val = int(n_samples * validation_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        for iteration in range(self.n_iterations):
            # Mini-batch gradient descent
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # Forward pass
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # Compute gradients
                dw = -(1/len(X_batch)) * np.dot(X_batch.T, (y_batch - y_pred))
                db = -(1/len(X_batch)) * np.sum(y_batch - y_pred, axis=0)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Print progress if requested
            if self.print_cost and iteration % 100 == 0:
                train_pred = np.dot(X_train, self.weights) + self.bias
                val_pred = np.dot(X_val, self.weights) + self.bias
                train_loss = np.mean((train_pred - y_train) ** 2)
                val_loss = np.mean((val_pred - y_val) ** 2)
                print(f"Iteration {iteration}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.dot(X, self.weights) + self.bias