import numpy as np

from LinearClassifier import LinearClassifier


class SoftmaxClassifier(LinearClassifier):
    def loss(self, X, y, reg):
        """
          Softmax loss function, vectorized version.

        """
        # Initialize the loss and gradient to zero.
        W = self.W
        num_train = X.shape[0]

        # calculate the loss
        scores = X * W
        scores -= np.max(scores, axis=1)
        scores = np.exp(scores)
        correct_class_score = scores[np.arange(len(scores)), y.T].T
        correct_class_score[correct_class_score == 0] = 0.000001
        loss = - np.sum(np.log(correct_class_score / np.sum(scores, axis=1))) / num_train

        # calculate the gradient
        props = scores / np.sum(scores, axis=1)
        z = np.zeros(props.shape)
        z[np.arange(len(scores)), y.T] = 1
        props = (props - z) / num_train
        dW = X.T * props

        # Regularization
        loss += 0.5 * reg * np.sum(np.multiply(W, W))
        dW += reg * W

        return loss, dW