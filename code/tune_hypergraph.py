import copy
import json
import numpy as np
from numpy import linalg as LA
from numpy.linalg import solve
import operator
import read_feature as rf
from util_io import calc_hyper_laplacian, read_historical_ranking, feature_scale

class tune_hypergraph:
    def __init__(self, model_para, feature_fnames_fname, final_list_fname, historical_ranking_list, hr_scale, hyper_k, channel):
        self.model_para = model_para
        self.final_list_fname = final_list_fname
        self.final_list = np.genfromtxt(final_list_fname, delimiter=',', dtype=str)
        self.feature_fnames_fname = feature_fnames_fname
        self.historical_ranking_list = historical_ranking_list
        self.hr_scale = hr_scale
        self.n = len(self.final_list)
        self.hyper_k = hyper_k
        self.X = self._read_single_channel_hyper_feature(channel)

        self.L = np.matrix(calc_hyper_laplacian(self.X, self.hyper_k))

        # read historical ranking
        self.y = np.matrix(read_historical_ranking(final_list=self.final_list, hr_fname=self.historical_ranking_list, scale=hr_scale)).T
        # print self.y.shape
        # for i in range(self.y.shape[1]):
        #     if self.y[0, i] > 1000:
        #         print self.final_list[i], self.y[0, i]

        # generated ranking
        # self.f = np.matrix(np.ones((self.n, 1)))
        self.f = np.matrix(np.arange(self.n)).T

    def ranking(self):
        print self._loss()
        b = self.model_para['lam_i'] * self.y
        self.f = np.matrix(solve(self.L + (self.model_para['lam_i']) * np.matrix(np.identity(self.n)), b))
        print 'final loss:', self._loss()
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

    def _loss(self):
        loss_gr = (0.5 * (self.f.T * self.L * self.f))[0, 0]
        loss_hr = (0.5) * self.model_para['lam_i'] * np.power(LA.norm(self.f - self.y), 2)
        return [loss_gr + loss_hr, loss_hr, loss_gr]

    def _read_single_channel_hyper_feature(self, channel):
        with open(self.feature_fnames_fname, 'r') as fin:
            feature_fnames = json.load(fin)
            fin.close()
            fnames = feature_fnames[channel]['text']
            print 'files selected:', fnames
            concatenated_feature = []
            is_first = True
            for fname in fnames:
                print 'reading feature from:', fname
                data = np.genfromtxt(fname, delimiter=',', dtype=float)
                print 'read data with shape:', data.shape
                if is_first:
                    is_first = False
                    concatenated_feature = copy.copy(data)
                else:
                    concatenated_feature = np.concatenate((concatenated_feature, data), axis=1)
                print 'current feature with shape:', concatenated_feature.shape
                print '-----------------------------------------------'
            feature_scale(concatenated_feature)
            return concatenated_feature
