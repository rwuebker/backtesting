import pandas as pd
import numpy as np
import cvxpy as cvx


class OptimalHoldings:

    def __init__(self, risk_model, alpha_vector, previous, risk_cap=0.05, factor_max=10.0, aum=50e6,
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
        self.aum = aum
        self.notional_risk = risk_cap * aum

    def _get_obj(self):
        ra = self.risk_aversion
        Q = self.risk_model.Q # Q is k x N
        i_var_vector = self.risk_model.i_var_vector
        ivv = cvx.Parameter((len(i_var_vector), 1))
        ivv.value = i_var_vector.values
        self.ivv = ivv
        av = self.alpha_vector.values.reshape(len(self.alpha_vector), 1)
        print('this is av.shape: ', av.shape)
        print('this is self.previous.shape: ', self.previous.shape)
        h0 = cvx.Parameter(len(self.previous))
        h0.value = self.previous
        lambda_vector = np.ones((len(av), 1)) * 0.1/2000000
        lv = cvx.Parameter(lambda_vector.shape)
        lv.value = lambda_vector
        h1 = cvx.Variable(len(self.alpha_vector))
        multiplier = cvx.Parameter(nonneg=True)
        multiplier.value = ra * 0.50
        func = 0.0
        func += multiplier * cvx.sum((Q @ h1) ** 2)
        func +=  multiplier * ((h1**2) @ ivv)
        func -= h1 @ av
        func += ((h1-h0) ** 2) @ lv
        m = ra * 0.50
        print(ivv.shape)
        print(h1.shape)
        #func = multiplier * cvx.sum((Q @ h1) ** 2) + multiplier * (h1**2) @ ivv - (h1 @ av) + ((h1-h0) ** 2) @ lv
        func = m * cvx.sum((Q @ h1)**2) + m * ((h1**2) @ ivv) - (h1 @ av) + ((h1-h0)**2) @ lv
        av = -av
        self.h1 = h1
        func = cvx.Minimize(av.T @ h1)
        self.obj_func = func

    def _get_constraints(self):
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
        return [risk <= self.risk_cap**2, factor_exposures.T @ h1 <= self.factor_max, factor_exposures.T @ h1 >= self.factor_min, \
                sum(h1) == 0, sum(cvx.abs(h1)) <= 1, h1 >= self.weights_min, h1 <= self.weights_max]

    def get_risk(self):
        Q = self.risk_model.Q
        h1 = self.h1
        ivv = self.ivv
        result = cvx.sum((Q @ h1)**2) + (h1**2) @ ivv
        ivm = self.risk_model.i_var_matrix.values
        f = self.risk_model.factor_exposures.T @ h1
        X = self.risk_model.factor_cov_matrix
        return cvx.quad_form(f, X) + cvx.quad_form(h1, ivm)

    def find(self):
        h1 = self.h1
        prob = cvx.Problem(self.obj_func, self._get_constraints())
        print(prob.is_dcp())
        prob.solve(max_iters=1000)
        optimal_weights = np.asarray(h1.value).flatten()
        df = pd.DataFrame(optimal_weights, index=self.alpha_vector.index)
        return df
