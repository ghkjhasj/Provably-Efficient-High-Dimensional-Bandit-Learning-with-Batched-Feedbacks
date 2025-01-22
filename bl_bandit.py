import random
import numpy as np
from numpy import linalg as la
from sklearn import linear_model as lm
import math
import time
from math import sqrt
from scipy import linalg
import torch


class BatchedLassoBandit(object):
    """
    Create a bandit object using batched lasso bandit algorithm
    """

    def __init__(self, D, K, h, t_0, lam_1, lam_2_0, c):
        """
        Initializes a lasso bandit contextual bandit object
        :param D: int, dimension of the context features
        :param K: int, number of arms
        :param h: float, localization parameter
        :param t_0: int, forced-sample parameter
        :param lam_1: float, initial regularization parameter 1
        :param lam_2_0: float, initial regularization parameter 2
        """

        self.D = D
        self.K = K
        self.h = h
        self.t_0 = t_0
        self.lam_1 = lam_1
        self.lam_2_0 = lam_2_0
        self.lam_2 = lam_2_0
        self.c = c

        self.r_est = [np.zeros((D, 1))] * K  # random-sample estimators
        self.w_est = [np.zeros((D, 1))] * K  # whole-sample estimators
        self.r_samp_idx = [[]] * K  # random sample index
        self.w_samp_idx = [[]] * K  # whole sample index
        # self.data = np.array([]).reshape(0, D)
        # self.reward = np.array([]).reshape(0, 1)

    def choose_grid(self, T, L):
        """
        Choose the grid before training
        :param T: int, total time steps
        :param L: int, number of batches
        :return: ndarray, indices of the grid
        """

        a = np.log(T) / np.log(L) * self.c
        print("a:%f", a)
        grid = [2]  # t_1
        for l in range(1, T + 1):
            val0 = grid[l - 1]
            val = (a / l + 1) * val0
            val = min(math.floor(val), T)
            grid.append(val)
            if val == T:
                break

        # grid = np.arange(110, T, 110)
        # grid = np.insert(grid, len(grid), T)
        return np.array(grid)

    def get_opt_action(self, x, t):
        """
        select the action
        :param x: D x 1 vector. D dimensional context feature vector
        :param t: Current time index
        :return: arm index for the current parameter and context set_with_var
        """

        d_t = np.random.binomial(1, np.min(np.array([1, self.t_0 / (t + 1)])))
        if d_t:
            a_t = random.sample(range(self.K), 1)[0]
            self.r_samp_idx[a_t] = np.append(self.r_samp_idx[a_t], t)
        else:
            K_hat = []
            for a in range(self.K):
                if x.T.dot(self.r_est[a]) > np.max(
                        [x.T.dot(self.r_est[a]) for a in range(self.K)]) - self.h / 2:
                    K_hat.append(a)
            p_t = np.zeros(self.K) - 1000
            for a in K_hat:
                p_t[a] = x.T.dot(self.w_est[a])

            a_t = np.argmax(p_t[K_hat])

        self.w_samp_idx[a_t] = np.append(self.w_samp_idx[a_t], t)
        return a_t

    def update_est(self, X, r, t_l):
        """
        updata the random and whole estimators at the end of l-th batch
        :param r_l: d_l list. rewards in batch l
        """

        self.lam_2 = self.lam_2_0 * np.sqrt((np.log(t_l) + np.log(self.D)) / t_l)
        for a in range(self.K):
            idx = np.array(self.r_samp_idx[a], dtype=int)
            if np.size(idx) > 0:
                reg_lasso_1 = lm.Lasso(alpha=self.lam_1, fit_intercept=True, max_iter=1000)
                reg_lasso_1.fit(X[idx, :], r[idx])
                self.r_est[a] = reg_lasso_1.coef_
                #self.r_est[a], _, _ = ista(X[idx, :], r[idx], self.lam_1, 10000)

            idx = np.array(self.w_samp_idx[a], dtype=int)
            if np.size(idx) > 0:
                reg_lasso_2 = lm.Lasso(alpha=self.lam_2, fit_intercept=True, max_iter=1000)
                reg_lasso_2.fit(X[idx, :], r[idx])
                self.w_est[a] = reg_lasso_2.coef_
                #self.w_est[a], _, _ = ista(X[idx, :], r[idx], self.lam_2, 10000)


class LassoBandit(object):
    """Creates a bandit object which uses the lasso bandit algorithm.
    (http://web.stanford.edu/~bayati/papers/LassoBandit.pdf)
    """

    def __init__(self, D, K, q, h, lambda_1, lambda_2_0):
        """Initializes a lasso bandit contextual bandit object
                Parameters
                ----------
                D : int
                    Dimension of the context features
                K : int
                    Number of arms
                q: int
                   forced sampling parameter
                h: float
                   localization parameter
                lambda_1: float
                    regularization parameter 1
                lambda_2_0: float
                    regularization parameter 2
        """

        self.D = D
        self.K = K
        self.q = np.int(q)
        self.h = h
        self.lambda_1 = lambda_1
        self.lambda_2_0 = lambda_2_0
        self.lambda_2 = lambda_2_0

        self.r_est = [np.zeros((D, 1))] * K  # random-sample estimators
        self.w_est = [np.zeros((D, 1))] * K  # whole-sample estimators
        self.r_samp_idx = [[]] * K  # random sample index
        self.w_samp_idx = [[]] * K  # whole sample index

        self.j_a = [np.arange(q * a + 1 - 1, q * (a + 1) + 1 - 1) for a in range(K)]
        self.n = 0

    def forced_sample_set(self, T):
        # initialize forced sample indices
        for a in range(self.K):
            base = 1
            finish = False
            while not finish:
                for j in range(int(self.q) * a + 1 - 1, int(self.q) * (a + 1) + 1 - 1):
                    idx = (base - 1) * self.K * self.q + j
                    if (idx < T):
                        self.r_samp_idx[a] = np.append(self.r_samp_idx[a], idx)
                    else:
                        finish = True
                        break
                base = base * 2
        self.r_samp_idx = np.array(self.r_samp_idx, dtype=int)

    def get_opt_action(self, x, t):
        """Calculates action.
                Parameters
                ----------
                x
                    D x 1 vector. D dimensional context feature vector
                t
                    Current time index
                Returns
                -------
                int
                    arm index for the current parameter and context set_with_var
                """

        # estimate!
        d_t = self.is_in_forced_sample_set(t)
        if d_t == -1:
            K_hat = []
            for a in range(self.K):
                if x.T.dot(self.r_est[a]) > np.max(
                        [x.T.dot(self.r_est[a]) for a in range(self.K)]) - self.h / 2:
                    K_hat.append(a)
            p_t = np.zeros(self.K)
            for a in K_hat:
                p_t[a] = x.T.dot(self.w_est[a])

            a_t = np.argmax(p_t[K_hat])

        else:
            a_t = d_t

        # self.data.append(x.reshape(1, self.D), axis=0)
        self.w_samp_idx[a_t] = np.append(self.w_samp_idx[a_t], t)
        return a_t

    def update_est(self, X, r, a_t, t):
        self.lambda_2 = self.lambda_2_0 * np.sqrt((np.log(t + 1) + np.log(self.D)) / (t + 1))

        idx = self.r_samp_idx[a_t]
        idx = idx[idx <= t]

        if np.size(idx) > 0 and self.is_in_forced_sample_set(t):
            reg_lasso_1 = lm.Lasso(alpha=self.lambda_1, fit_intercept=False, max_iter=100000)
            reg_lasso_1.fit(X[idx, :], r[idx])
            self.r_est[a_t] = reg_lasso_1.coef_
            #self.r_est[a_t], _, _ = ista(X[idx, :], r[idx], self.lambda_1, 10000)

        idx = np.array(self.w_samp_idx[a_t], dtype=int)
        if np.size(idx) > 0:
            reg_lasso_2 = lm.Lasso(alpha=self.lambda_2, fit_intercept=False, max_iter=100000)
            reg_lasso_2.fit(X[idx, :], r[idx])
            self.w_est[a_t] = reg_lasso_2.coef_
            #self.w_est[a_t], _, _ = ista(X[idx, :], r[idx], self.lambda_2, 10000)

    def is_in_forced_sample_set(self, t):
        """Checks if the time index is in the forced sample set_with_var.
                    Parameters
                    ----------
                    t: int
                        Current time index
                    Returns
                    -------
                    a_out: int
                        arm index for the corresponding forced sample set_with_var.
                            -1 if not in forced sample set_with_var
                    """

        a_out = -1
        for a in range(self.K):
            if np.isin(t, self.r_samp_idx[a]):
                a_out = a

        return a_out


# Pricing
class Price_LassoBandit(object):
    """
    Creates a bandit object for pricing which uses the lasso bandit algorithm.
    """

    def __init__(self, D, q, h, lambda_2_0, device):
        """
        Initializes a lasso bandit contextual bandit object
                Parameters
                ----------
                D : int
                    Dimension of the context features
                lambda_2_0: float
                    regularization parameter 2
        """

        self.D = D
        self.q = q
        self.h = h
        self.lambda_2_0 = lambda_2_0
        self.lambda_2 = lambda_2_0
        self.device = device

        self.beta_est = np.zeros((D, 1))  # estimators
        self.r_samp_idx = [[]] * 2  # random sample index

    def forced_sample_set(self, T):
        # initialize forced sample indices
        base = 1
        finish = False
        while not finish:
            for a in range(2):
                idx = base ** 2 + a - 1
                if (idx < T):
                    self.r_samp_idx[a] = np.append(self.r_samp_idx[a], idx)
                else:
                    finish = True
                    break
            base = base + 1

        self.r_samp_idx = np.array(self.r_samp_idx, dtype=int)

    def get_opt_action(self, x, t):
        """
            Calculates action.
            Parameters
            ----------
            x: D x 1 vector. D dimensional context feature vector
            t: Current time index
            Returns
            -------
            int, arm index for the current parameter and context set_with_var
        """

        # estimate!
        d_t = self.is_in_forced_sample_set(t)
        if d_t == -1:
            d = x.shape[0]
            if x.T.dot(self.beta_est[d:]):
                p_t = -0.5 * (x.T.dot(self.beta_est[:d])) / (x.T.dot(self.beta_est[d:]))
                p_t = min(max(p_t[0], 0), 1000)
            else:
                p_t = 0
        elif d_t == 0:
            p_t = 200
        else:
            p_t = 600
        return p_t

    def update_est(self, X, y, p_t, t):
        self.lambda_2 = self.lambda_2_0 * np.sqrt((np.log(t) + np.log(self.D)))
        XX = np.concatenate((X, np.multiply(X.T, p_t).T), axis=1)

        """# OLS
        if np.linalg.det((XX.T).dot(XX)):
            self.beta_est = np.linalg.inv((XX.T).dot(XX) + np.diag(np.ones(XX.shape[1]) * self.lambda_2)).dot((XX.T).dot(y.reshape(-1, 1)))"""

        reg_lasso_2 = lm.Lasso(alpha=self.lambda_2, fit_intercept=False, max_iter=1e6)
        reg_lasso_2.fit(XX, y)
        self.beta_est = reg_lasso_2.coef_

        """XX = torch.tensor(XX, device=self.device)
        y = torch.tensor(y, device=self.device)
        self.beta_est, _, _ = ista(XX, y, torch.tensor(self.lambda_2, dtype=float, device=self.device), 10000, self.device)"""

    def is_in_forced_sample_set(self, t):
        """
        Checks if the time index is in the forced sample set_with_var.
            Parameters
            ----------
                t: int, Current time index
            Returns
            -------
                a_out: int, arm index for the corresponding forced sample set_with_var.
                -1 if not in forced sample set_with_var
        """

        a_out = -1
        for a in range(2):
            if np.isin(t, self.r_samp_idx[a]):
                a_out = a
        return a_out


class Price_BatchedLassoBandit(object):
    """
    Create a bandit object for pricing using batched lasso bandit algorithm
    """

    def __init__(self, D, t_0, lam_2_0, c, device):
        """
        Initializes a lasso bandit contextual bandit object
        :param D: int, dimension of the context features
        :param t_0: int, forced-sample parameter
        :param lam_2_0: float, initial regularization parameter 2
        :param c: float, control batch number
        :param device: device
        """

        self.D = D
        self.t_0 = t_0
        self.lam_2_0 = lam_2_0
        self.lam_2 = lam_2_0
        self.c = c
        self.device = device

        self.beta_est = np.zeros((D, 1))  # whole-sample estimators

    def choose_grid(self, T, L):
        """
        Choose the grid before training
        :param T: int, total time steps
        :param L: int, number of batches
        :return: ndarray, indices of the grid
        """

        a = np.log(T) / np.log(L) * self.c
        print("a:%f", a)
        grid = [2]  # t_1
        for l in range(1, T + 1):
            val0 = grid[l - 1]
            val = (a / l + 1) * val0
            val = min(math.floor(val), T)
            grid.append(val)
            if val == T:
                break
        return np.array(grid)

    def get_opt_action(self, x, t):
        """
            Calculates action.
            Parameters
            ----------
            x: D x 1 vector. D dimensional context feature vector
            t: Current time index
            Returns
            -------
            int, arm index for the current parameter and context set_with_var
        """

        d_t = np.random.binomial(1, np.min(np.array([1, self.t_0 / (t + 1)])))
        if d_t:
            p_t = random.sample([200, 600], 1)[0]
        else:
            d = x.shape[0]
            if x.T.dot(self.beta_est[d:]):
                p_t = -0.5 * (x.T.dot(self.beta_est[:d])) / (x.T.dot(self.beta_est[d:]))
                p_t = min(max(p_t[0], 0), 1000)
            else:
                p_t = 0
        return p_t

    def update_est(self, X, y, p_l, t_l):
        """
        updata the estimators at the end of l-th batch
        :param X: t_l x D, the contexts up to the end of t_l
        :param y: t_l , rewards in batch l
        :param p_l: t_l, price up to the end of t_l
        :param t_l: time step at the end of batch l
        """

        self.lam_2 = self.lam_2_0 * math.pow(t_l, 0.25) * np.sqrt((np.log(t_l) + np.log(self.D)))
        XX = np.concatenate((X, np.multiply(X.T, p_l).T), axis=1)

        """# OLS
        if np.linalg.det((XX.T).dot(XX)):
            self.beta_est = np.linalg.inv((XX.T).dot(XX) + np.diag(np.ones(XX.shape[1]) * self.lam_2)).dot(
                (XX.T).dot(y.reshape(-1, 1)))"""

        reg_lasso_2 = lm.Lasso(alpha=self.lam_2, fit_intercept=False, max_iter=1e6)
        reg_lasso_2.fit(XX, y)
        self.beta_est = reg_lasso_2.coef_

        """XX = torch.tensor(XX, device=self.device)
        y = torch.tensor(y, device=self.device)
        self.beta_est, _, _ = ista(XX, y, torch.tensor(self.lam_2, dtype=float, device=self.device), 10000, self.device)"""


# LASSO Estimator
def soft_thresh(x, l, device='cpu'):
    return torch.sign(x) * torch.maximum(torch.abs(x) - l, torch.zeros(x.shape, device=device))


def ista(A, b, l, maxit, device='cpu'):
    TOL = 1e-4
    A = A.to(device).to(torch.float)
    b = b.to(device).to(torch.float)
    l = l.to(device).to(torch.float)
    l = l.to(torch.float)
    x = torch.zeros((A.shape[1], 1), device=device, dtype=torch.float)
    pobj = []
    L = torch.norm(A) ** 2  # Lipschitz constant
    L = L.to(torch.float).to(device)
    time0 = time.time()
    for _ in range(maxit):
        x_n = soft_thresh(x + torch.mm(A.T, b - torch.mm(A, x)) / L, l / L, device)
        this_pobj = 0.5 * torch.norm(torch.mm(A, x) - b) ** 2 + l * torch.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))
        if torch.norm(x_n - x, np.inf) <= TOL:
            break
        x = x_n

    times, pobj = map(torch.tensor, zip(*pobj))
    if device == 'cuda':
        x = x.cpu().numpy()
    return x[:, 0], pobj, times


def fista(A, b, l, maxit):
    TOL = 1e-5
    x = np.zeros((A.shape[1], 1))
    pobj = []
    t = 1
    z = x.copy()
    L = linalg.norm(A) ** 2
    time0 = time.time()
    for _ in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x_n = soft_thresh(z, l / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        z = x_n + ((t0 - 1.) / t) * (x_n - xold)
        this_pobj = 0.5 * linalg.norm(A.dot(x_n) - b) ** 2 + l * linalg.norm(x_n, 1)
        pobj.append((time.time() - time0, this_pobj))
        if la.norm(x_n - x, np.inf) <= TOL:
            break
        x = x_n

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times
