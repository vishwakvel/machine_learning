import numpy as np

class LinearRegressionNE: # Normal Equation: y = Xw + b -> w* = (X^T * X)^-1 * X^T * y
    def __init__(self):
        self.weights = None
    
    def add_bias_column(self, X):
        rows = X.shape[0] # getting the number of rows in the matrix X
        ones = np.ones((rows, 1)) # creating a column of ones with the same number of rows as X
        return np.concatenate([ones, X], axis=1) # concatenating the ones column at thebeginning of matrix X
    
    def fit(self, X, y):
        X_b = self.add_bias_column(X) # adding bias column to the matrix X
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y # solving the normal equation
        return self

    def predict(self, X):
        X_b = self.add_bias_column(X)
        return X_b @ self.weights # matrix multiplication of X (with bias column) and the weights
    
    def score(self, X, y):
        y_predicted = self.predict(X)
        ss_residual = np.sum((y - y_predicted) ** 2) # sum of squared residuals
        ss_total = np.sum((y - y.mean()) ** 2) # sum of squared total
        return 1 - (ss_residual / ss_total) # R-squared score

class LinearRegressionGD: # Gradient Descent: w* = w - a * ∇L(w) -> w = w - a * (2/n) * X^T * (Xw - y)
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []
    
    def add_bias_column(self, X):
        rows = X.shape[0] # getting the number of rows in the matrix X
        ones = np.ones((rows, 1)) # creating a column of ones with the same number of rows as X
        return np.concatenate([ones, X], axis=1) # concatenating the ones column at thebeginning of matrix X
    
    def fit(self, X, y):
        X_b = self.add_bias_column(X)
        n_samples = X_b.shape[0] # number of rows in the matrix X_b
        self.weights = np.zeros(X_b.shape[1]) # initializing the weights to zero
        self.loss_history = [] # initializing the loss history to an empty list
        
        for _ in range(self.n_iterations): # solving the gradient descent equation
            y_predicted = X_b @ self.weights
            error = y_predicted - y
            gradient = (2 / n_samples) * X_b.T @ error
            self.weights -= self.learning_rate * gradient

            loss = np.mean(error ** 2) # mean squared error
            self.loss_history.append(loss)

        return self
    
    def predict(self, X):
        X_b = self.add_bias_column(X)
        return X_b @ self.weights # matrix multiplication of X (with bias column) and the weights
    
    def score(self, X, y):
        y_predicted = self.predict(X)
        ss_residual = np.sum((y - y_predicted) ** 2) # sum of squared residuals
        ss_total = np.sum((y - y.mean()) ** 2) # sum of squared total
        return 1 - (ss_residual / ss_total) # R-squared score