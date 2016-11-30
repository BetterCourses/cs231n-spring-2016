import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    adapted from https://goo.gl/zfRtLr

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

    num_classes = W.shape[0]
    num_train = X.shape[0]

    for i in range(num_train):
        # Compute vector of scores
        f_i = W.dot(X[i, :]) # in R^{num_classes}

        # Normalization trick to avoid numerical instability
        # per http://cs231n.github.io/linear-classify/#softmax
        log_c = np.max(f_i)
        f_i -= log_c

        # Compute loss (and add to it, divided later)
        # L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
        sum_i = 0.0
        for f_i_j in f_i:
            sum_i += np.exp(f_i_j)
        loss += -f_i[y[i]] + np.log(sum_i)

        # Compute gradient
        # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
        # Here we are computing the contribution to the inner sum for a given i.
        for j in range(num_classes):
            p = np.exp(f_i[j])/sum_i
            dW[j, :] += (p-(j == y[i])) * X[i, :]

    # Compute average
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W


    return loss, dW


# http://cs231n.github.io/neural-networks-case-study/#grad
# http://cs231n.github.io/optimization-2/#mat
def softmax_loss_vectorized(W, X, y, reg, debug=False):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    For numerical stability:
    we should shift the values inside the vector score so that the
    highest value in each row is zero
    """
    N = X.shape[0]

    # compute class score
    scores = X.dot(W.T)  # (N, C)

    # numerical stable implementation
    scores -= np.amax(scores, axis=1).reshape(N, 1)

    e_fk = np.exp(scores)  # (N, C)

    e_fsum = e_fk.sum(axis=1)  # sum of all exp score, (N, )

    # copmute class probabilities
    p_k = e_fk / e_fsum.reshape(N, 1)  # (N, C)

    # compute the loss
    p_yi = p_k[np.arange(N), y]
    loss = np.mean(-np.log(p_yi))
    loss += 0.5 * reg * np.sum(W * W)  # L2 regularization


    # backprop pass
    dscores = p_k
    dscores[np.arange(N), y] -= 1
    dscores /= N

    # since score = X @ W.T
    # now we have gradient back -> dscore, we need to times it to local gradient
    # which is d (X @ W.T) / d W -> X
    # so work on dscore @ X, do the dimension analysis
    # then we figure out it should be dscore.T @ X

    dW = np.dot(dscores.T, X)

    # regularization backprop contribution to gradient
    dW += reg * W

    if debug:
        print('e_fk shape = {}'.format(e_fk.shape))
        print('p_k shape = {}'.format(p_k.shape))
        print('dscores shape = {}'.format(dscores.shape))
        print('dW shape = {}'.format(dW.shape))

    return loss, dW
