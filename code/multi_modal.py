import copy
import numpy as np
from numpy import linalg as LA
from numpy.linalg import solve
import operator
import read_feature as rf
from util_io import calc_conv_laplacian, read_historical_ranking

class multi_modal:
    def __init__(self, model_para, feature_fnames_fname, final_list_fname, historical_ranking_list, hr_scale):
        self.model_para = model_para
        self.final_list_fname = final_list_fname
        self.final_list = np.genfromtxt(final_list_fname, delimiter=',', dtype=str)
        self.feature_fnames_fname = feature_fnames_fname
        self.historical_ranking_list = historical_ranking_list
        self.hr_scale = hr_scale
        reader = rf.feature_reader(self.feature_fnames_fname, self.final_list_fname, read_type=2)
        self.n = len(self.final_list)
        self.feature_names, self.Xs = reader.read_feature()
        self.S = len(self.feature_names)
        # # TEST
        # print self.X.shape
        # print self.X[0][:10]
        # print self.X[0][self.X.shape[1] - 10:]
        # print self.X[self.X.shape[0] - 1][:10]
        # print self.X[self.X.shape[0] - 1][self.X.shape[1] - 10:]

        self.Ls = []
        for i in range(self.S):
            self.Ls.append(np.matrix(calc_conv_laplacian(self.Xs[i])))
        # # TEST
        # print self.L.shape
        # print self.L[0][:10]
        # print self.L[0][self.L.shape[1] - 10:]
        # print self.L[self.L.shape[0] - 1][:10]
        # print self.L[self.L.shape[0] - 1][self.L.shape[1] - 10:]

        # read historical ranking
        self.y = np.matrix(read_historical_ranking(final_list=self.final_list, hr_fname=self.historical_ranking_list, scale=hr_scale)).T
        # print self.y.shape
        # for i in range(self.y.shape[1]):
        #     if self.y[0, i] > 1000:
        #         print self.final_list[i], self.y[0, i]

        # generated ranking
        # self.f = np.matrix(np.ones((self.n, 1)))
        self.f = np.matrix(np.arange(self.n)).T
        self.alpha = np.empty((self.S, 1))
        self.alpha.fill(1.0 / self.S)
        self.alpha = np.matrix(self.alpha)

    def ranking(self):
        self._optimize()
        rank_pos_pair = {}
        for ind in range(self.n):
            rank_pos_pair[ind] = self.f[ind, 0]
        rank_sorted = sorted(rank_pos_pair.items(), key=operator.itemgetter(1))
        generated_ranking = {}
        cur_rank = 0
        for ind, rs_kv in enumerate(rank_sorted):
            name = self.final_list[rs_kv[0]]
            if ind == 0:
                pre_rank = -100
            else:
                pre_rank = rank_sorted[ind - 1][1]
            if abs(rs_kv[1] - pre_rank) > self.model_para['thres']:
                cur_rank += 1
            generated_ranking[name] = cur_rank
        # for ind, name in enumerate(self.final_list):
        #     generated_ranking[name] = self.f[ind, 0]
        return generated_ranking

    def _calc_alpha(self):
        # print 'before optimization, beta: ', self.alpha
        A = np.matrix(np.zeros((self.S + 1, self.S + 1)))
        t = np.matrix(np.ones((self.S + 1, 1)))

        # calculate t
        for i in range(self.S):
            t[i, 0] = self.f.T * self.Ls[i] * self.f * (-0.5)

        # calculate A
        for i in range(self.S):
            A[i, self.S] = -1
            A[self.S, i] = 1
            A[i, i] = self.model_para['lam_r']
            # for j in range(i, self.S):
            #     if i == j:
            #         A[i, j] = T[i].T * T[i] * (1.0 / self.N) + self.lam3
            #     else:
            #         A[i, j] = T[i].T * T[j] * (1.0 / self.N)
            #         A[j, i] = A[i, j]

        # calculate alpha
        # solved_alpha = A.I * t
        solved_alpha = solve(A, t)
        print 'solved alpha:', solved_alpha
        for i in range(self.S):
            self.alpha[i, 0] = solved_alpha[i, 0]
        print 'current alpha:', self.alpha
    def _calc_f(self):
        late_L = np.matrix(np.zeros((self.n, self.n)))
        for i in range(self.S):
            late_L += self.alpha[i, 0] * self.Ls[i]
        self.f = np.matrix(solve(late_L + (self.model_para['lam_i']) * np.matrix(np.identity(self.n)), self.model_para['lam_i'] * self.y))

    def _loss(self):
        loss_gr = 0.0
        for i in range(self.S):
            loss_gr += (0.5 * (self.f.T * self.Ls[i] * self.f))[0, 0] * self.alpha[i, 0]
        loss_hr = (0.5) * self.model_para['lam_i'] * np.power(LA.norm(self.f - self.y), 2)
        loss_reg = 0.5 * self.model_para['lam_r'] * np.power(LA.norm(self.alpha), 2)
        return [loss_gr + loss_hr + loss_reg, loss_hr, loss_gr, loss_reg]

    def _optimize(self):
        loss = self._loss()
        print 'init loss:', loss
        iter_count = 0
        while True:
            self._calc_f()
            cur_loss = self._loss()
            print 'loss after f:', cur_loss
            self._calc_alpha()
            cur_loss = self._loss()
            print 'loss after alpha:', cur_loss
            # cur_loss = self._loss()
            if abs(cur_loss[0] - loss[0]) < 0.001:
                print 'converge'
                break
            loss = copy.copy(cur_loss)
            if iter_count > 20:
                print 'arrive max iteration'
                break
            iter_count += 1
        print 'final loss:', loss