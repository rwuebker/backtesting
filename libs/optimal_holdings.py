import pandas as pd
import numpy as np
import cvxpy as cvx

class OptimalHoldingsCVX:

    def __init__(self, risk_model, alpha_vector, previous, risk_cap=0.05, factor_max=10.0, aum=50e6, lambda_reg=0.50,
                 factor_min=-10.0, weights_max=0.015, weights_min=-0.015, risk_aversion=1.0e-6):

        self.risk_model = risk_model
        self.alpha_vector = alpha_vector
        self.previous = previous
        self.risk_cap = risk_cap
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.weights_min = weights_min
        self.weights_max = weights_max
        self.risk_aversion = risk_aversion
        self.aum = aum
        self.notional_risk = risk_cap * aum
        self.lambda_reg = lambda_reg

    def _get_obj(self):
        start = time.time()
        ra = self.risk_aversion
        Q = self.risk_model.Q # Q is k x N
        ivv = self.risk_model.i_var_vector
        av = self.alpha_vector.values.reshape(len(self.alpha_vector), 1)
        h0 = self.previous
        lambda_vector = np.ones((len(av), 1)) * 0.1/2000000
        lv = lambda_vector
        h1 = cvx.Variable(len(self.alpha_vector))
        m = ra * 0.50
        #func = m * cvx.sum((Q @ h1)**2) + m * ((h1**2) @ ivv) - (h1 @ av) + ((h1-h0)**2) @ lv + self.lambda_reg * cvx.norm(h1, 2)
        func = -(h1 @ av) + self.lambda_reg * cvx.norm(h1, 2)
        func = cvx.Minimize(func)
        self.is_dcp = cvx.Problem(func, []).is_dcp()
        self.h1 = h1
        self.obj_func = func
        end = time.time()
        return

    def _get_constraints(self):
        start = time.time()
        risk = self.get_risk()
        c = []
        h1 = self.h1
        factor_exposures = self.risk_model.factor_exposures
        c.append(risk <= self.risk_cap ** 2)
        c.append(h1 <= self.weights_max)
        c.append(h1 >= self.weights_min)
        c.append(sum(h1) == 0)
        c.append(factor_exposures.T @ h1 <= self.factor_max)
        c.append(factor_exposures.T @ h1 >= self.factor_min)
        c.append(sum(cvx.abs(h1)) <= 1.0)
        end = time.time()
        return c

    def get_risk(self):
        start = time.time()
        Q = self.risk_model.Q
        h1 = self.h1
        ivv = self.risk_model.i_var_vector
        result = cvx.sum((Q @ h1)**2) + (h1**2) @ ivv
        ivm = self.risk_model.i_var_matrix.values
        f = self.risk_model.factor_exposures.T @ h1
        X = self.risk_model.factor_cov_matrix
        end = time.time()
        return cvx.quad_form(f, X) + cvx.quad_form(h1, ivm)

    def find(self):
        self._get_obj()
        print(self.is_dcp)
        h1 = self.h1
        prob = cvx.Problem(self.obj_func, self._get_constraints())
        start = time.time()
        print(prob.is_dcp())
        prob.solve(max_iters=25)
        optimal_weights = np.asarray(h1.value).flatten()
        df = pd.DataFrame(optimal_weights, index=self.alpha_vector.index)
        end = time.time()
        return df
