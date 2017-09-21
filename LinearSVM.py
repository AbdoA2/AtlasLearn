import numpy as np

from LinearClassifier import LinearClassifier


class LinearSVM(LinearClassifier):
    def loss(self, X, y, reg):
        """
          Structured SVM loss function, vectorized implementation.
          X - Input examples (N, D)
          y - if given is (N, 1) used for calculating dW
          reg - regularization constant
        """
        W = self.W
        num_train = X.shape[0]

        # calculate loss
        scores = X * W
        class_score = scores[np.arange(len(scores)), y.T].T
        scores = scores - class_score + 1
        scores[np.arange(len(scores)), y.T] = 0
        loss = np.sum(scores[scores > 0]) / num_train
        loss += 0.5 * reg * np.sum(np.multiply(W, W))

        # calculate the gradient
        m = (scores > 0).astype(np.int32)
        m[np.arange(len(scores)), y.T] = - np.sum(m, axis=1)[np.arange(len(scores))].T
        dW = X.T * m
        dW /= num_train
        dW += reg * W

        return loss, dW