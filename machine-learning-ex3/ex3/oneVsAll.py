import scipy.optimize as opt
import lrCostFunction as lCF
from sigmoid import *


def one_vs_all(X, y, num_labels, lmd):
    # Some useful variables
    (m, n) = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data 2D-array
    X = np.c_[np.ones(m), X]

    for i in range(num_labels):
        print('Optimizing for handwritten number {}...'.format(i))
        # ===================== Your Code Here =====================
        # Instructions : You should complete the following code to train num_labels
        #                logistic regression classifiers with regularization
        #                parameter lambda
        #
        #
        # Hint: you can use y == c to obtain a vector of True(1)'s and False(0)'s that tell you
        #       whether the ground truth is true/false for this class
        #
        # Note: For this assignment, we recommend using opt.fmin_cg to optimize the cost
        #       function. It is okay to use a for-loop (for c in range(num_labels) to
        #       loop over the different classes
        #
        initial_theta = np.zeros((n + 1, 1))
        iClass = i if i else 10
        y_i = np.array([1 if x == iClass else 0 for x in y])

        def cost_func(t):
            return lCF.lr_cost_function(t, X, y_i, lmd)[0]

        def grad_func(t):
            return lCF.lr_cost_function(t, X, y_i, lmd)[1]

        theta, *unused = opt.fmin_cg(cost_func, x0=initial_theta, fprime=grad_func, maxiter=100, disp=False, full_output=True)
        all_theta[i] = theta


        # ============================================================    
        print('Done')

    return all_theta
