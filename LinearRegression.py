import numpy as np


class LinearRegression:
    def __init__(self, input_dim):
        self.W = 0.001 * np.random.rand(input_dim, 1)
        self.b = 0

    def loss(self, X, y, reg=0):
        W = self.W
        l = X * W + self.b - y
        loss = 0.5 * np.mean(np.square(l))
        dW = np.mean(np.multiply(l, X), axis=0).T
        db = np.mean(l)

        # Regularization
        loss += 0.5 * reg * np.sum(np.multiply(W, W))
        dW += reg * W

        return loss, dW, db

    def train(self, X, y, reg=0, num_iters=200, batch_size=100, learning_rate=0.01, verbose=False):
        loss_history = []
        for it in range(num_iters):
            indices = np.random.choice(X.shape[0], size=batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient
            loss, dW, db = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W -= learning_rate * dW
            self.b -= learning_rate * db

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history