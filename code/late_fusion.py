import copy
import numpy as np
from numpy import linalg as LA
from numpy.linalg import solve
import operator
import read_feature as rf
from util_io import calc_conv_laplacian, read_historical_ranking

class late_fusion:
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
        self.fs = []
        for i in range(self.S):
            self.fs.append(np.matrix(np.random.rand(self.n, 1)))

    def ranking(self):
        self._optimize()
        late_f = np.matrix(np.zeros((self.n, 1)))
        for i in range(self.S):
            late_f += self.fs[i]
        late_f *= 0.2
        rank_pos_pair = {}
        for ind in range(self.n):
            rank_pos_pair[ind] = late_f[ind, 0]
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

    # def _calc_f(self):
    #     for i in range(self.S):
    #         b = np.matrix(np.zeros((self.n, 1)))
    #         for j in range(self.S):
    #             if not j == i:
    #                 b += self.fs[j]
    #         b = (self.y - b * 0.2) * (self.model_para['lam_i'] * 0.2)
    #         self.fs[i] = np.matrix(solve(self.Ls[i] + (self.model_para['lam_i'] * 0.04) * np.matrix(np.identity(self.n)), b))
    #         # print self._loss()
    def _calc_f(self):
        for i in range(self.S):
            # b = np.matrix(np.zeros((self.n, 1)))
            # for j in range(self.S):
            #     if not j == i:
            #         b += self.fs[j]
            b = (self.y) * (self.model_para['lam_i'])
            # self.fs[i] = np.matrix(solve(self.Ls[i] + (self.model_para['lam_i'] * 0.04) * np.matrix(np.identity(self.n)), b))
            self.fs[i] = np.matrix(solve(self.Ls[i] + (self.model_para['lam_i']) * np.matrix(np.identity(self.n)), b))
            # print self._loss()

    def _loss(self):
        loss_gr = 0.0
        for i in range(self.S):
            loss_gr += (0.5 * (self.fs[i].T * self.Ls[i] * self.fs[i]))[0, 0]
        late_f = np.matrix(np.zeros((self.n, 1)))
        for i in range(self.S):
            late_f += self.fs[i]
        late_f *= 0.2
        loss_hr = (0.5) * self.model_para['lam_i'] * np.power(LA.norm(late_f - self.y), 2)
        return [loss_gr + loss_hr, loss_hr, loss_gr]

    def _optimize(self):
        loss = self._loss()
        print 'init loss:', loss
        iter_count = 0
        while True:
            self._calc_f()
            cur_loss = self._loss()
            if abs(cur_loss[0] - loss[0]) < 0.001:
                print 'converge at iteration:', iter_count
                break
            loss = copy.copy(cur_loss)
            if iter_count > 200:
                print 'arrive max iteration'
                break
            iter_count += 1
        print 'final loss:', loss