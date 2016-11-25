import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (C, D) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    For numerical stability:
    we should shift the values inside the vector score so that the
    highest value in each row is zero
    """
    dW = np.zeros_like(W)
    N = X.shape[0]

    score = X.dot(W.T)  # (N, C)

    # numerical stability implementation
    score -= np.amax(score, axis=1).reshape(N, 1)

    exp_score = np.exp(score)

    ef_yi = exp_score[np.arange(N), y]  # right class score, (N, )
    assert ef_yi.shape == (N, )

    ef_sum = exp_score.sum(axis=1)  # sum of all class, (N, )
    assert ef_sum.shape == (N, )

    p_yi = ef_yi / ef_sum

    loss = np.mean(-np.log(p_yi))
    loss += 0.5 * reg * np.sum(W * W)  # regularization


    return loss, dW
