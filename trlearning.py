# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.linalg import norm, block_diag
from scipy.sparse.linalg import eigsh
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class TransferLearning:
    def __init__(self, k: int, **kwargs):
        self.matrices = dict()
        self.params = {
            'model': kwargs.pop('model', 'JDA'),
            'kernel': kwargs.pop('kernel', 'primal'),
            'lambda': kwargs.pop('lambda', 0.1),
            'k': k,
            'iter_lim': kwargs.pop('iter_lim', 100),
            'criteria': kwargs.pop('criteria', 0.1),
            'Cs': kwargs.pop('Cs', 0.01),
            'Ct': kwargs.pop('Ct', 0.01),
            'classification': kwargs.pop('classification', True),
            'activation': kwargs.pop('activation', 'sigmoid'),
            'NL': kwargs.pop('NL', 0),
            'threshold': kwargs.pop('threshold', 0.5),
            'stop_iter': kwargs.pop('stop_iter', 5)
        }

        if self.params['model'] not in ('JDA', 'DTELM'):
            raise ValueError('Only JDA and DTELM models are selectable!')
        if not isinstance(self.params['classification'], bool):
            raise ValueError('Please identify if this problem is '
                             'classification or not!')
        if self.params['activation'] not in ('sigmoid', 'sin'):
            raise ValueError('Only sigmoid and sin activation functions '
                             'are selectable!')

        # TODO: Currently there is only one kernel.
        if self.params['kernel'] != 'primal':
            self.params['kernel'] = 'primal'
        # TODO: Currently only classification is accepted.
        if not self.params['classification']:
            self.params['classification'] = True

    def _preprocessing(self, x_s, x_t, y_s, amt_features):
        if x_s.shape[0] != y_s.shape[0]:
            raise ValueError('Inconsistent sample number between feature data'
                             'and target lables!')
        if x_s.shape[1] != x_t.shape[1]:
            raise ValueError('Inconsistent feature number between the source '
                             'and the target domain!')
        if ((isinstance(x_s, pd.DataFrame)
             and not isinstance(x_t, pd.DataFrame))
                or (not isinstance(x_s, pd.DataFrame)
                    and isinstance(x_t, pd.DataFrame))):
            raise TypeError('Inconsistent data type between the source '
                            'and the target domain!')
        elif (isinstance(x_s, pd.DataFrame) and isinstance(x_t, pd.DataFrame)
              and x_s.columns.tolist() != x_t.columns.tolist()):
            raise ValueError('Different features between the source '
                             'and the target domain!')
        # Train a baseline model to predict y_t
        lr_model, y_t = self._train_predict_model(x_s, x_t, y_s, amt_features)

        if isinstance(x_s, pd.DataFrame):
            x_s = x_s.values.T.copy()
        elif isinstance(x_s, np.ndarray):
            x_s = x_s.T.copy()
        else:
            raise TypeError('Invalid data type!')

        if isinstance(x_t, pd.DataFrame):
            x_t = x_t.values.T.copy()
        elif isinstance(x_t, np.ndarray):
            x_t = x_t.T.copy()
        else:
            raise TypeError('Invalid data type!')

        if isinstance(y_s, pd.Series):
            y_s = y_s.values.copy()
        elif isinstance(y_s, np.ndarray):
            y_s = y_s.copy()
        else:
            raise TypeError('Invalid data type!')

        return x_s, x_t, y_s, y_t, lr_model

    def _train_predict_model(self, x_s, x_t, y_s, amt_features):
        df_x_s = pd.DataFrame(x_s)
        df_x_t = pd.DataFrame(x_t)
        n_s = len(df_x_s.index)
        n_t = len(df_x_t.index)
        feature_types = df_x_s.dtypes
        features = {
            'cat': feature_types.loc[feature_types == np.dtype('O')].index.tolist(),
            'num': feature_types.loc[feature_types != np.dtype('O')].index.tolist(),
            'amt': amt_features
        }
        if (set(np.intersect1d(features['amt'], features['num']))
                != set(features['amt'])):
            raise ValueError('Invalid amount features!')
        features['num'] = [x for x in features['num']
                           if x not in features['amt']]

        imputer = {'num': SimpleImputer(strategy='median'),
                   'amt': SimpleImputer(strategy='constant', fill_value=0),
                   'cat': SimpleImputer(strategy='constant', fill_value='')}
        oh_encoder = OneHotEncoder(sparse=False, dtype=np.int,
                                   handle_unknown='ignore')
        scaler = MinMaxScaler()
        std_data = dict()
        for ft in ('num', 'amt', 'cat'):
            if len(features[ft]) > 0:
                std_data.update({
                    ft: {
                        's': pd.DataFrame(
                            imputer[ft].fit_transform(df_x_s.loc[:, features[ft]]),
                            columns=features[ft]),
                        't': pd.DataFrame(
                            imputer[ft].fit_transform(df_x_t.loc[:, features[ft]]),
                            columns=features[ft])
                    }
                })
            else:
                std_data.update({
                    ft: {'s': pd.DataFrame(index=list(range(n_s))),
                         't': pd.DataFrame(index=list(range(n_t)))}
                })
        onehots = {'s': pd.DataFrame(index=list(range(n_s))),
                   't': pd.DataFrame(index=list(range(n_t)))}
        for feat in features['cat']:
            for d in ('s', 't'):
                onehot_array = oh_encoder.fit_transform(
                    std_data['cat'][d].loc[:, [feat, ]])
                enum_cats = oh_encoder.categories_[0].tolist()
                onehot_array = np.delete(onehot_array, enum_cats.index(''),
                                         axis=1)
                del enum_cats[enum_cats.index('')]
                enum_cats = [str(feat) + '_' + x for x in enum_cats]
                onehots[d] = pd.concat(
                    [onehots[d], pd.DataFrame(onehot_array, columns=enum_cats)],
                    axis=1, ignore_index=True)
        x_source = pd.concat(
            [std_data['num']['s'], std_data['amt']['s'], onehots['s']],
            axis=1, ignore_index=True).values
        x_target = pd.concat(
            [std_data['num']['t'], std_data['amt']['t'], onehots['t']],
            axis=1, ignore_index=True).values
        x_source = scaler.fit_transform(x_source)
        x_target = scaler.fit_transform(x_target)

        lr_model = LogisticRegression()
        lr_model.fit(x_source, y_s)
        y_pred_proba = lr_model.predict_proba(x_target)[:, 1]
        y_pred = np.where(y_pred_proba < self.params['threshold'], 0, 1)
        return lr_model, y_pred

    @staticmethod
    def normalise_vectors(x):
        return np.dot(x, np.diag(1. / np.sqrt(np.sum(x ** 2, axis=0))))

    def fit(self, x_s, x_t, y_s, amt_features: list):
        if (self.params['model'] == 'JDA'
                and self.params['k'] >= x_s.shape[1] - 1):
            raise ValueError('The number of target domain dimensions is '
                             'too large!')

        x_s, x_t, y_s, y_t, y_model = self._preprocessing(x_s, x_t, y_s,
                                                          amt_features)

        if not any(y_s):
            raise ValueError('Invalid target labels!')

        x = TransferLearning.normalise_vectors(np.hstack((x_s, x_t)))
        m, n = x.shape
        n_s = x_s.shape[1]
        n_t = x_t.shape[1]
        n_classes = len(np.unique(y_s))

        f1_prev = -np.inf
        same_count = 0
        f1 = 0
        iter_count = 0
        while (iter_count < self.params['iter_lim']
               and 1 - f1 >= self.params['criteria']
               and same_count < self.params['stop_iter']):
            e = np.vstack([1. / n_s * np.ones((n_s, 1)),
                           -1. / n_t * np.ones((n_t, 1))])
            self.matrices.update({'M': np.dot(e, e.T) * n_classes})
            for c in np.unique(y_s):
                e1 = np.zeros((n_s, 1))
                e1[y_s == c] = 1. / sum(y_s == c)
                e2 = np.zeros((n_t, 1))
                e2[y_t == c] = -1. / sum(y_t == c)
                e = np.vstack((e1, e2))
                e[np.isinf(e)] = 0
                self.matrices['M'] += np.dot(e, e.T)
            self.matrices['M'] = (self.matrices['M'] / norm(self.matrices['M'],
                                                            ord='fro'))
            if self.params['model'] == 'JDA':
                # Construct centering matrix
                self.matrices.update({
                    'H': np.eye(n) - 1. / (n * np.ones((n, n)))
                })

                if self.params['kernel'] == 'primal':
                    self.matrices.update({
                        'A': eigsh((np.dot(np.dot(x, self.matrices['M']), x.T)
                                    + self.params['lambda'] * np.eye(m)),
                                   k=self.params['k'],
                                   M=np.dot(np.dot(x, self.matrices['H']), x.T),
                                   which='SM')[1]
                    })
                else:
                    self.matrices.update({'A': None})

                lr_model, y_pred_t = self._train_predict_model(
                    np.dot(self.matrices['A'].T, x_s).T,
                    np.dot(self.matrices['A'].T, x_t).T,
                    y_s, amt_features)
                y_pred_s = lr_model.predict_proba(
                    np.dot(self.matrices['A'].T, x_s).T)[:, 1]
                y_pred_s = np.where(y_pred_s < self.params['threshold'], 0, 1)
                f1 = f1_score(np.hstack((y_s, y_t)),
                              np.hstack((y_pred_s, y_pred_t)))
                if f1 == f1_prev:
                    same_count += 1
                f1_prev = f1
                y_t = y_pred_t
                iter_count += 1
            else:
                input_weights = np.random.random((self.params['k'], m)) * 2. - 1.
                bias = np.random.random((self.params['k'], 1))
                if self.params['activation'] == 'sigmoid':
                    self.matrices.update({
                        'Hs': 1. / (1. + np.exp(-(np.dot(input_weights, x_s)
                                                  + np.tile(bias, (1, n_s))))),
                        'Ht': 1. / (1. + np.exp(-(np.dot(input_weights, x_t)
                                                  + np.tile(bias, (1, n_t)))))
                    })
                else:
                    self.matrices.update({
                        'Hs': np.sin(np.dot(input_weights, x_s)
                                     + np.tile(bias, (1, n_s))),
                        'Ht': np.sin(np.dot(input_weights, x_t)
                                     + np.tile(bias, (1, n_t)))
                    })
                self.matrices.update({
                    'H': np.hstack((self.matrices['Hs'],
                                    self.matrices['Ht'])).T
                })
                self.matrices['Hs'] = self.matrices['Hs'].T
                self.matrices['Ht'] = self.matrices['Ht'].T
                alias = {'A': None, 'B': None, 'Binv': None, 'C': None, 'D': None,
                         'ApT': None, 'ApS': None}
                if self.params['NL'] == 0:
                    alias['A'] = np.dot(self.matrices['Ht'],
                                        self.matrices['Hs'].T)
                    alias['B'] = (np.dot(self.matrices['Ht'],
                                         self.matrices['Ht'].T)
                                  + np.eye(n_t) / self.params['Ct'])
                    alias['Binv'] = pinv(alias['B'])
                    alias['C'] = np.dot(self.matrices['Hs'],
                                        self.matrices['Ht'].T)
                    alias['D'] = (np.dot(self.matrices['Hs'],
                                         self.matrices['Hs'].T)
                                  + np.eye(n_s) / self.params['Cs'])
                    alias['ApT'] = (np.dot(alias['Binv'], y_t)
                                    - np.dot(np.dot(np.dot(alias['Binv'],
                                                           alias['A']),
                                                    pinv(np.dot(
                                                        np.dot(alias['C'],
                                                               alias['Binv']),
                                                        alias['A']) - alias['D'])),
                                             np.dot(np.dot(alias['C'],
                                                           alias['Binv']),
                                                    y_t) - y_s))
                    alias['ApS'] = np.dot(pinv(np.dot(np.dot(alias['C'],
                                                             alias['Binv']),
                                                      alias['A']) - alias['D']),
                                          np.dot(np.dot(alias['C'], alias['Binv']),
                                                 y_t) - y_s)
                    output_weights = (np.dot(self.matrices['Hs'].T, alias['ApS'])
                                      + np.dot(self.matrices['Ht'].T, alias['ApT']))
                else:
                    output_weights = np.dot(pinv(np.eye(self.params['k'])
                                                 + np.dot(np.dot(self.params['Cs'],
                                                                 self.matrices['Hs'].T),
                                                          self.matrices['Hs'])
                                                 + np.dot(np.dot(self.params['Ct'],
                                                                 self.matrices['Ht'].T),
                                                          self.matrices['Ht'])),
                                            (np.dot(np.dot(self.params['Cs'],
                                                           self.matrices['Hs'].T),
                                                    y_s)
                                             + np.dot(np.dot(self.params['Ct'],
                                                             self.matrices['Ht'].T),
                                                      y_t)))
                y_pred_s = np.dot(self.matrices['Hs'], output_weights).T
                y_pred_s = np.where(y_pred_s < self.params['threshold'], 0, 1)
                y_pred_t = np.dot(self.matrices['Ht'], output_weights).T
                y_pred_t = np.where(y_pred_t < self.params['threshold'], 0, 1)
                f1 = f1_score(np.hstack((y_s, y_t)),
                              np.hstack((y_pred_s, y_pred_t)))
                if f1 == f1_prev:
                    same_count += 1
                f1_prev = f1
                y_t = y_pred_t
                iter_count += 1
        if self.params['model'] == 'DTELM':
            mdelta = block_diag(np.eye(n_s) * self.params['Cs'],
                                np.eye(n_t) * self.params['Ct'])
            m_delta_m = mdelta + np.dot(self.params['lambda'],
                                        self.matrices['M'])
            if n < self.params['k']:
                self.matrices.update({
                    'A': np.dot(np.dot(np.dot(self.matrices['H'].T,
                                              pinv(np.ones((n, n))
                                                   + np.dot(np.dot(m_delta_m,
                                                                   self.matrices['H']),
                                                            self.matrices['H'].T))),
                                       mdelta),
                                x.T).T
                })
            else:
                self.matrices.update({
                    'A': np.dot(np.dot(np.dot(
                        pinv(np.ones((self.params['k'], self.params['k']))
                             + np.dot(np.dot(self.matrices['H'].T, m_delta_m),
                                      self.matrices['H'])),
                        self.matrices['H'].T), mdelta), x.T).T
                })

    def transform(self, x):
        return TransferLearning.normalise_vectors(
            np.dot(self.matrices['A'].T, x.T))

    def fit_transform(self, x_s, x_t, y_s, y_t):
        self.fit(x_s, x_t, y_s, y_t)
        return self.transform(x_s), self.transform(x_t)


if __name__ == '__main__':
    from sklearn.datasets import load_wine

    wine = load_wine()
    data = wine['data']
    y = wine['target']
    data_1 = data[y != 2]
    data_2 = data[y != 1]
    y_1 = y[y != 2]
    y_2 = y[y != 1]
    y_2[y_2 > 0] = 1
    tr = TransferLearning(model='DTELM', k=10, criteria=0.3)
    tr.fit(data_1, data_2, y_1, [])
