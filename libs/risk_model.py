import pandas as pd
import numpy as np
import datetime as dt
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
        var_s = s.var(ddof=1) * self.ann_factor
        self.i_var_matrix = pd.DataFrame(data=np.diag(var_s), index=self.returns.columns,
                                                     columns=self.returns.columns)
        self.i_var_vector = pd.DataFrame(data=var_s, index=self.returns.columns)

    def predict_portfolio_risk(self, weights):
        weights = np.asarray(weights)



    @staticmethod
    def winz(x, a=-0.10, b=0.10):
        return np.where(x <= a, a, np.where(x >= b, b, x))

    def _winsorize_returns(self):
        self.unwinz_returns = self.returns
        self.returns = self.returns.apply(RiskModelPCA.winz)



