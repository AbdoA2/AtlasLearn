import numpy as np


def affine_forward(X, W, b):
    out = X * W + b
    cache = (X, W, b)

    return out, cache


def relu_forward(X):
    out = X.copy()
    out[X < 0] = 0

    return out, X


def affine_backward(dout, cache, calc_dx=True):
    X, W, b = cache
    dW = X.T * dout
    db = np.sum(dout, axis=0)
    if calc_dx:
        dX = dout * W.T
    else:
        dX = None

    return dX, dW, db


def relu_backward(dout, cache):
   X = cache
   dX = dout.copy()
   dX[X < 0] = 0

   return dX


def affine_relu_forward(X, W, b):
    out1, fc_cache = affine_forward(X, W, b)
    out, relu_cache = relu_forward(out1)

    return out, (fc_cache, relu_cache)


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    dout1 = relu_backward(dout, relu_cache)
    dX, dW, db = affine_backward(dout1, fc_cache)

    return dX, dW, db