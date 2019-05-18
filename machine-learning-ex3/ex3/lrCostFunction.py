import numpy as np
from sigmoid import *


def lr_cost_function(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    hypothesis = sigmoid(np.dot(X, theta)) # a^Tb = b^Ta, if a and b are vectors.
    reg_theta = theta[1:]
    cost = np.sum(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)) / m \
           + (lmd / (2 * m)) * np.sum(theta**2)
    error = hypothesis - y
    grad = np.dot(X.T, error) / m
    grad[1:] = grad[1:] + (lmd / m) * reg_theta   
    # =========================================================

    return cost, grad
