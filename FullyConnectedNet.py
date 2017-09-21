import numpy as np
from utils.layers import *
from utils.layers_utils import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
           dropout=0, use_batchnorm=False, reg=0.0,
           weight_scale=1e-4, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
        the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
        initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
        this datatype. float32 is faster but less accurate, so you should use
        float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
        will make the dropout layers deteriminstic so we can gradient check the
        model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        layrs_size = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            n = str(i+1)
            self.params['W' + n] = weight_scale * np.random.rand(layrs_size[i], layrs_size[i+1])
            self.params['b' + n] = weight_scale * np.random.rand(1, layrs_size[i+1])
            if use_batchnorm and i < self.num_layers - 1:
                self.params['gamma' + n] = np.ones((1, layrs_size[i+1]))
                self.params['beta' + n] = np.zeros((1, layrs_size[i+1]))

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            for i in range(self.num_layers - 1):
                self.bn_params += [{'mode':'train'}]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None, reg=0.0):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None

        outs, caches = [0] * (self.num_layers+1), [0] * (self.num_layers+1)
        outs[0] = X
        dropout_caches = [0] * (self.num_layers+1)

        # Propagate through the first L-1 layers
        for i in range(1, self.num_layers):
            n = str(i)
            if self.use_batchnorm:
                outs[i], caches[i] = affine_norm_relu_forward(outs[i-1], self.params['W' + n], self.params['b' + n],
                                                            self.params['gamma' + n], self.params['beta' + n],
                                                            self.bn_params[i-1])
            else:
                outs[i], caches[i] = affine_relu_forward(outs[i-1], self.params['W' + n], self.params['b' + n])

            if self.use_dropout:
                outs[i], dropout_caches[i] = dropout_forward(outs[i], self.dropout_param)

        # The final layer
        n = str(self.num_layers)
        outs[-1], caches[-1] = affine_forward(outs[-2], self.params['W' + n], self.params['b' + n])
        scores = outs[-1]

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        douts = [0] * (self.num_layers+1)
        douts[-1], grads['W' + n], grads['b' + n] = affine_backward(dscores, caches[-1])
        for i in range(self.num_layers - 1, 0, -1):
            n = str(i)
            if self.use_dropout:
                douts[i+1] = dropout_backward(douts[i+1], dropout_caches[i])

            if self.use_batchnorm:
                douts[i], grads['W' + n], grads['b' + n], grads['gamma' + n], grads['beta' + n] \
                    = affine_norm_relu_backward(douts[i+1], caches[i], calc_dx=(i != 1))
            else:
                douts[i], grads['W' + n], grads['b' + n] \
                    = affine_relu_backward(douts[i+1], caches[i], calc_dx=(i != 1))

        # Regularization
        if reg > 0:
            for i in range(1, self.num_layers+1):
                loss += 0.5 * reg * np.sum(np.multiply(self.params['W' + n], self.params['W' + n]))
                grads['W' + n] += reg * self.params['W' + n]

        return loss, grads

    def train(self, X, y, X_val, y_val,
        learning_rate=1e-3, learning_rate_decay=0.95,
        reg=1e-5, num_iters=100,
        batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
        X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
        after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # Sample mini batch
            indices = np.random.choice(X.shape[0], size=batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # Update the weights
            for i in range(1, self.num_layers+1):
                n = str(i)
                self.params['W' + n] -= learning_rate * grads['W' + n]
                self.params['b' + n] -= learning_rate * grads['b' + n]
                if self.use_batchnorm and ('gamma' + n) in grads:
                    self.params['gamma' + n] -= learning_rate * grads['gamma' + n]
                    self.params['beta' + n] -= learning_rate * grads['beta' + n]

            # Check for verbose
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
        classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
        the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
        to have class c, where 0 <= c < C.
        """
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred