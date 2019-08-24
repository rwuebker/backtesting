import pandas as pd
import numpy as np
import datetime as dt
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA


class RiskModelPCA:

    def __init__(self, returns, n_components=20, window=365, ann_factor=252):
        self.returns = returns
        self.n_components = n_components
        self.window = window
        self.ann_factor = ann_factor
        self.calc_data()

    def calc_data(self):
        self._winsorize_returns()
        self._fit_pca()
        self._create_factor_names()
        self._calc_factor_exposures()
        self._calc_factor_returns()
        self._calc_factor_cov_matrix()
        self._calc_idiosyncratic_var()
        self._calc_matrices()


    def _fit_pca(self):
        pca = PCA(n_components=self.n_components, svd_solver='full')
        pca.fit(self.returns)
        self.pca = pca

    def _create_factor_names(self):
        self.factor_names = ['factor_{}'.format(i) for i in range(self.n_components)]

    def _calc_factor_exposures(self):
        self.factor_exposures =  pd.DataFrame(data=self.pca.components_.T, index=self.returns.columns,
                                              columns=self.factor_names)

    def _calc_factor_returns(self):
        self.factor_returns = pd.DataFrame(data=self.pca.transform(self.returns), index=self.returns.index,
                                           columns=self.factor_names)

    def _calc_factor_cov_matrix(self):
        var_ = self.factor_returns.var(ddof=1) * self.ann_factor
        self.factor_cov_matrix = np.diag(var_)

    def _calc_idiosyncratic_var(self):
        common_returns = self.factor_returns.dot(self.factor_exposures.T)
        s = self.returns - common_returns
        var_s = np.var(s) * self.ann_factor
        self.i_var_matrix = pd.DataFrame(data=np.diag(var_s), index=self.returns.columns,
                                                     columns=self.returns.columns)
        self.i_var_vector = pd.DataFrame(data=var_s, index=self.returns.columns)

    def _calc_matrices(self):
        # we can remove any N by N calculations using the following
        # common risk is usually h.T * B * F * B.T * h
        # letting G = sqrt(F):  h.T * B * G*G * B.T * h
        # letting Q.T = B * G and Q = G * B.T :  h.T * Q.T * Q * h
        self.F = self.factor_cov_matrix
        self.G = sqrtm(self.F)
        self.B = self.factor_exposures
        self.Q = self.G.dot(self.B.T)
        # will use these to calculate risk in predict_portfolio_risk_opt

    def predict_portfolio_risk(self, holdings):
        holdings = np.asarray(holdings)
        factor_var = np.add(self.B.dot(self.F.dot(self.B.T)), self.i_var_matrix)
        result = holdings.T.dot(factor_var.dot(holdings))
        return result

    def predict_portfolio_risk_opt(self, holdings):
        holdings = np.asarray(holdings)
        R = np.matmul(self.Q, holdings)
        common_risk = np.sum(R ** 2)
        h2 = holdings ** 2
        spec_risk = np.dot(h2, self.i_var_vector)
        result = common_risk + spec_risk
        return result[0]

    @staticmethod
    def winz(x, a=-0.10, b=0.10):
        return np.where(x <= a, a, np.where(x >= b, b, x))

    def _winsorize_returns(self):
        self.unwinz_returns = self.returns
        self.returns = self.returns.apply(RiskModelPCA.winz)



