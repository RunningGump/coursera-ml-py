import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]

    # Return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned logistic regression parameters.
    #                You should set p to a 1D-array of 0's and 1's
    #
    p = sigmoid(np.dot(X, theta))
    pos_idx = np.where(p >= 0.5)
    neg_idx = np.where(p < 0.5)

    p[pos_idx] = 1
    p[neg_idx] = 0

    # ===========================================================

    return p
