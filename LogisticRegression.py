import numpy as np


class LogisticRegression:
    def __init__(self, input_dim):
        self.W = 0.000 * np.random.randn(input_dim, 1)
        self.b = 0

    def loss(self, X, y, reg=0):
        W = self.W
        h = 1 / (np.exp(-X * W - self.b) + 1)
        z = 1 - h
        z[z == 0] = 0.000001
        h[h == 0] = 0.000001
        loss = - 0.5 * np.mean(np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(z)))
        dW = np.mean(np.multiply(h - y, X), axis=0).T
        db = np.mean(h)

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