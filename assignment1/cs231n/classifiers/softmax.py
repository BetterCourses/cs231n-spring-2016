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

# http://cs231n.github.io/neural-networks-case-study/#grad
def softmax_loss_vectorized(W, X, y, reg, debug=False):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    For numerical stability:
    we should shift the values inside the vector score so that the
    highest value in each row is zero
    """
    dW = np.zeros_like(W)
    N = X.shape[0]

    f_k = X.dot(W.T)  # (N, C)

    # numerical stability implementation
    f_k -= np.amax(f_k, axis=1).reshape(N, 1)

    e_fk = np.exp(f_k)  # (N, C)

    e_fsum = e_fk.sum(axis=1)  # sum of all exp score, (N, )

    p_k = e_fk / e_fsum.reshape(N, 1)  # (N, C)

    p_yi = p_k[np.arange(N), y]


    loss = np.mean(-np.log(p_yi))
    loss += 0.5 * reg * np.sum(W * W)  # regularization

    dscores = p_k
    dscores[np.arange(N), y] -= 1
    dscores /= N

    dW = dscores.T.dot(X)
    dW += reg * W

    if debug:
        print('e_fk shape = {}'.format(e_fk.shape))
        print('p_k shape = {}'.format(p_k.shape))
        print('dscores shape = {}'.format(dscores.shape))
        print('dW shape = {}'.format(dW.shape))

    
    return loss, p_k
