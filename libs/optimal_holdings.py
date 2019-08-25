import pandas as pd
import numpy as np
import cvxpy as cvx


class OptimalHoldingsCVX:

    def __init__(self, risk_model, alpha_vector, previous, risk_cap=0.05, factor_max=10.0, gmv=5e3, lambda_reg=0.00,
                 factor_min=-10.0, weights_max=0.50, weights_min=-0.50, risk_aversion=1.0e-6):

        self.risk_model = risk_model
        self.alpha_vector = alpha_vector
        self.previous = previous
        self.risk_cap = risk_cap
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.weights_min = weights_min * gmv
        self.weights_max = weights_max * gmv
        self.risk_aversion = risk_aversion
        self.gmv = gmv
        self.lambda_reg = lambda_reg

    def _get_obj(self, h1):
        ra = self.risk_aversion
        Q = self.risk_model.Q # Q is k x N
        ivv = self.risk_model.i_var_vector
        av = -self.alpha_vector
        h0 = self.previous.flatten()
        lv = np.ones((len(av), 1)) * 0.1/25000000
        #obj_func = cvx.Minimize(np.array(av.T) * h1 + self.lambda_reg * cvx.norm(h1, 2) + (h1-h0)**2 @ lv)
        obj_func = cvx.Minimize(np.array(av.T) * h1 + self.lambda_reg * cvx.norm(h1, 2))
        return obj_func

    def _get_constraints(self, h1, risk):
        c = []
        factor_exposures = self.risk_model.factor_exposures
        c.append(risk <= self.risk_cap**2)
        #c.append(factor_exposures.T @ h1 <= self.factor_max)
        #c.append(factor_exposures.T @ h1 >= self.factor_min)
        c.append(sum(h1) == 0)
        c.append(sum(cvx.abs(h1)) <= self.gmv)
        c.append(h1 >= self.weights_min)
        c.append(h1 <= self.weights_max)
        return c

    def get_risk(self, h1):
#         av_index = self.alpha_vector.index
#         Q = self.risk_model.Q
#         ivv = self.risk_model.i_var_vector
#         risk = cvx.sum((Q @ (h1 / self.gmv))**2) + ((h1/self.gmv)**2) @ ivv
#         return risk
        av_index = self.alpha_vector.index
        ivm = self.risk_model.i_var_matrix.loc[av_index].values
        f = self.risk_model.factor_exposures.loc[av_index].values.T @ h1
        X = self.risk_model.factor_cov_matrix
        risk = cvx.quad_form(f, X) + cvx.quad_form(h1/self.gmv, ivm)
        return risk

    def find(self):
        h1 = cvx.Variable(len(self.alpha_vector))
        risk = self.get_risk(h1)
        obj_func = self._get_obj(h1)
        constraints = self._get_constraints(h1, risk)
        prob = cvx.Problem(obj_func, constraints)
        prob.solve(max_iters=1000, verbose=True)
        optimal_weights = np.asarray(h1.value).flatten()
        df = pd.DataFrame(optimal_weights, index=self.alpha_vector.index, columns=['holdings'])
        return df
