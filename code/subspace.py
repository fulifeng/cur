import copy
import numpy as np
from numpy import linalg as LA
from numpy.linalg import solve
import operator
import read_feature as rf
from util_io import calc_conv_laplacian, read_historical_ranking

class subspace:
    def __init__(self, model_para, feature_fnames_fname, final_list_fname, historical_ranking_list, result_fname, evaluator, hr_scale):
        self.model_para = model_para
        self.final_list_fname = final_list_fname
        self.final_list = np.genfromtxt(final_list_fname, delimiter=',', dtype=str)
        self.feature_fnames_fname = feature_fnames_fname
        self.historical_ranking_list = historical_ranking_list
        self.hr_scale = hr_scale
        self.results_fname = result_fname
        self.evaluator = evaluator
        reader = rf.feature_reader(self.feature_fnames_fname, self.final_list_fname, read_type=2)
        self.n = len(self.final_list)
        # self.feature_names, self.Xs = reader.read_feature()
        # self.S = len(self.feature_names)
        #
        # self.Ls = []
        # for i in range(self.S):
        #     self.Ls.append(np.matrix(calc_conv_laplacian(self.Xs[i])))
        #
        # # read historical ranking
        # self.y = np.matrix(read_historical_ranking(final_list=self.final_list, hr_fname=self.historical_ranking_list, scale=hr_scale)).T

    def ranking(self):
        # read a list of rankings and find the best one, each ranking is wrote in a line
        data = np.genfromtxt(self.results_fname, delimiter=',')
        print 'data with shape:', data.shape, 'from file:', self.results_fname
        best_ranking = {}
        best_micra_f1 = 0.0
        for i in range(data.shape[0]):
            late_f = np.matrix(data[i]).T
            # print late_f.shape
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
            per = self.evaluator.evaluate(generated_ranking)
            # print 'current performance:', per
            if per['mic_f'] > best_micra_f1:
                best_micra_f1 = per['mic_f']
                best_ranking = copy.copy(generated_ranking)
                # print 'better performance:', per
        return best_ranking

    def write_for_matlab(self):
        for i in range(self.S):
            np.savetxt('subspace_learning/' + self.feature_names[i] + '.csv', self.Xs[i], delimiter=',')
        np.savetxt('subspace_learning/y.csv', self.y, delimiter=',')

