import numpy as np

class LogisticRegression: # w = w - a * ∇L(w) -> w = w - a * (1/n) * X^T * (y_predicted - y)
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []
    
    def add_bias_column(self, X):
        rows = X.shape[0] # getting the number of rows in the matrix X
        ones = np.ones((rows, 1)) # creating a column of ones with the same number of rows as X
        return np.concatenate([ones, X], axis=1) # concatenating the ones column at thebeginning of matrix X

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) # sigmoid function
    
    def fit(self, X, y):
        X_b = self.add_bias_column(X)
        n_samples = X_b.shape[0] # number of rows in the matrix X_b
        self.weights = np.zeros(X_b.shape[1]) # initializing the weights to zero
        self.loss_history = [] # initializing the loss history to an empty list

        for _ in range(self.n_iterations): # solving the logistic regression equations
            z = X_b @ self.weights # Logit (linear combination of features and weights) = ln(p / (1 - p)) where p is the probability of the positive class (y = 1)
            y_predicted = self.sigmoid(z)
            error = y_predicted - y
            gradient = (1 / n_samples) * X_b.T @ error
            self.weights -= self.learning_rate * gradient

            loss = -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15)) # log loss (1e-15 to avoid log(0))
            self.loss_history.append(loss)

        return self
    
    def predict(self, X):
        X_b = self.add_bias_column(X)
        probabilities = self.sigmoid(X_b @ self.weights)
        return (probabilities >= 0).astype(int) # 0 is the threshold for classification which can be adjusted as needed (higher threshold means more conservative classification)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) # accuracy score