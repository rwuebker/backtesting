import pandas as pd
import numpy as np
import cvxpy as cvx


class OptimalHoldings:

    def __init__(self, risk_model, alpha_vector, previous, risk_cap=0.05, factor_max=10.0,
                 factor_min=-10.0, weights_max=0.02, weights_min=-0.02, risk_aversion=1.0e-6):

        self.risk_model = risk_model
        self.alpha_vector = alpha_vector
        self.previous = previous
        self.risk_cap = risk_cap
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.weights_min = weights_min
        self.weights_max = weights_max
        self.risk_aversion = risk_aversion

    def _get_obj(self):
        ra = self.risk_aversion
        Q = self.risk_model.Q
        i_var_vector = self.risk_model.i_var_vector
        av = self.alpha_vector
        h0 = self.previous
        lambda_vector = np.ones((len(av), 1)) * 0.1/2000000
        h1 = cvx.Variable(len(av))
        func = 0.0
        func += 0.50 * ra * np.sum(np.matmul(Q, h1) ** 2)
        func +=  0.50 * ra * np.dot(h1**2, i_var_vector)
        func -= np.dot(h1, av)
        func += np.dot((h1-h0) ** 2, lambda_vector)
        func = cvx.Minimize(func)
