import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[0]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W.T)
        correct_class_score = scores[y[i]]
        diff_cnt = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                
                # if margin > 0, this is where gradient happens
                diff_cnt += 1
                dW[j,:] += X[i] # sums each contribution of the x_i's
        dW[y[i], :] -= diff_cnt * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    return loss, dW

# note
# 1. numpy array index trick: https://goo.gl/eqU8Yy
# 2. SVM loss fn: http://cs231n.github.io/linear-classify/#svm

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    #############################################################################
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    m, n = X.shape
    delta = 1

    all_score = np.dot(X, W.T)  # (m, 10)

    # 1. for each row, get the right score from y
    # 2. reshape it as column vector
    right_class_score = all_score[np.arange(m), y].reshape(m, 1)

    # use broadcast to minus the right score + delta, the right class will be 1 here
    margin = all_score - right_class_score + delta

    # shouldn't count the right class
    margin[np.arange(m), y] = 0

    thresh = np.maximum(0, margin)  # (m, 10)

    loss = thresh.sum() / m

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    binary = thresh > 0  # (m, 10)
    counting = binary.sum(axis=1)

    # this way, the right class will have it's right negative counting, and the rest are 0/1
    binary[np.arange(m), y] = -counting

    # this is the matrix math I dont fully comprehend, feels right
    dW = np.dot(binary.T, X) / m

    dW += reg * W

    return loss, dW
