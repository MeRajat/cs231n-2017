import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
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
  # pass
  num_samples = X.shape[0]
  num_classes = W.shape[1]
  loss = 0.0
  for i in xrange(num_samples):
      # compute scores
      score_i = X[i].dot(W)

      # numerical stability
      score_i -= np.max(score_i)

      # compute loss
      denominator = np.sum(np.exp(score_i))  # sum
      p = lambda k:np.exp(score_i[k])/ denominator
      loss += -np.log(np.exp(score_i[y[i]])/ denominator)

      # now gradient computation
      for j in xrange(num_classes):
          prob_j = p(j)  # probability of j class
          dW[:, j] += (prob_j - (j == y[i])) * X[i]

        #   if i == j:
        #       dW[:, j] +=  X[i] * ( prob_j - 1)
        #   else:
        #       dW[:, j] += X[i] * prob_j
  # loss normalization
  loss /= num_samples
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_samples
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_sample = X.shape[0]
  score = X.dot(W)
  score -= np.matrix(np.max(score, axis = 1)).T

  all_denominators =  np.sum(np.exp(score), axis=1, keepdims=True)
  values = np.exp(score) / all_denominators
  loss = np.sum(-np.log(values[np.arange(num_sample), y]))

  # calculate gradient
  # create empty array like values and assign 1 to where correct class for that
  # example
  tmp = np.zeros_like(values)
  tmp[np.arange(num_sample), y]  = 1
  dW = X.T.dot(values - tmp) # multiply as tmp => Sj and X => Si

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  loss /= num_sample
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_sample
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
