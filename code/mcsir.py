import copy
import json
import numpy as np
from numpy import linalg as LA
from numpy.linalg import solve
import operator
import read_feature as rf
from util_io import calc_conv_laplacian, read_historical_ranking

class mcsir:
    def __init__(self, model_para, clusters_fnames_fname, final_list_fname, historical_ranking_list, confidence_hrl, hr_scale):
        self.model_para = model_para
        print 'parameters:', self.model_para
        self.final_list_fname = final_list_fname
        self.final_list = np.genfromtxt(final_list_fname, delimiter=',', dtype=str)
        self.clusters_fnames_fname = clusters_fnames_fname
        self.historical_ranking_list = historical_ranking_list
        self.confidence_hrl = confidence_hrl
        self.hr_scale = hr_scale
        # reader = rf.feature_reader(self.feature_fnames_fname, self.final_list_fname, read_type=1)
        self.n = len(self.final_list)
        # self.X = reader.read_feature()

        # self.L = np.matrix(calc_conv_laplacian(self.X))
        self.L_clusters = []
        self.S_clusters = []
        self.alpha_clusters = []
        self.clusters = 0
        self._read_laplacian_matrices_by_cluster()

        self.beta = np.empty((self.clusters, 1))
        self.beta.fill(1.0 / self.clusters)
        self.beta = np.matrix(self.beta)
        self.f_clusters = []
        for i in range(self.clusters):
            # self.f_clusters.append(np.matrix(np.arange(self.n)).T)
            self.f_clusters.append(np.matrix(np.random.rand(self.n, 1)))

        # read historical ranking
        self.y = np.matrix(read_historical_ranking(final_list=self.final_list, hr_fname=self.historical_ranking_list, scale=hr_scale)).T
        self.C = np.matrix(np.diag(read_historical_ranking(final_list=self.final_list, hr_fname=self.confidence_hrl, scale=hr_scale)))

        # generated ranking
        # self.f = np.matrix(np.ones((self.n, 1)))

    def ranking(self):
        self._optimize()
        rank_pos_pair = {}
        late_f = np.matrix(np.zeros((self.n, 1)))
        for i in range(self.clusters):
            late_f += self.f_clusters[i] * self.beta[i, 0]
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

    def ranking_list(self):
        self._optimize()
        rank_pos_pair = {}
        late_f = np.matrix(np.zeros((self.n, 1)))
        for i in range(self.clusters):
            late_f += self.f_clusters[i] * self.beta[i, 0]
        for ind in range(self.n):
            rank_pos_pair[ind] = late_f[ind, 0]
        rank_sorted = sorted(rank_pos_pair.items(), key=operator.itemgetter(1))
        rank_score = []
        for i in range(self.n):
            rank_score.append([self.final_list[rank_sorted[i][0]], rank_sorted[i][1]])
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
        # rank_score = []
        rank_index = []
        for i in range(self.n):
            # rank_score.append([self.final_list[i], late_f[i]])
            rank_index.append([rank_score[i][0], generated_ranking[rank_score[i][0]]])
        return rank_score, rank_index

    def write_parameters(self):
        # write fs
        # with open('./final_parameter/fs.csv', 'w') as fin:
        self._optimize()
        concatenated_fs = np.matrix(np.ones((self.n, self.clusters), dtype=float))
        for i in range(self.clusters):
            concatenated_fs[:, i] = copy.copy(self.f_clusters[i][:, 0])
        np.savetxt('./final_parameter/fs.csv', concatenated_fs, delimiter=',')

        # write beta
        np.savetxt('./final_parameter/beta.csv', self.beta, delimiter=',')

        # write alpha
        with open('./final_parameter/alpha.csv', 'w') as fout:
            for i in range(self.clusters):
                if not i == 0:
                    fout.write('\n')
                # fout.write(','.join(self.alpha_clusters[i]))
                for j in range(self.alpha_clusters[i].shape[0]):
                    fout.write(str(self.alpha_clusters[i][j, 0]) + ',')

        # write Laplacian matrix
        for i in range(self.clusters):
            L_common = np.matrix(np.zeros((self.n, self.n), dtype=float), dtype=float)
            for j in range(self.alpha_clusters[i].shape[0]):
                L_common += self.alpha_clusters[i][j, 0] * self.L_clusters[i][j]
            np.savetxt('./final_parameter/laplacian_' + str(i) + '.csv', L_common, delimiter=',')

    def _calc_alpha(self):
        for i in range(self.clusters):
            if self.alpha_clusters[i].shape[0] == 1:
                self.alpha_clusters[i] = np.matrix([1.0])
                # print 'unchanged:', self.alpha_clusters[i]
            else:
                self._calc_alpha_cluster(i)
                # self._calc_alpha_cluster_new(i)

    # def _calc_alpha_cluster_new(self, k):
    #     # print 'before optimization, beta: ', self.alpha
    #     size_cluster = self.alpha_clusters[k].shape[0]
    #     A = np.matrix(np.zeros((size_cluster, size_cluster)))
    #     t = np.matrix(np.ones((size_cluster, 1)))
    #
    #     # calculate t
    #     for i in range(size_cluster):
    #         t[i, 0] = (self.f_clusters[k].T * self.L_clusters[k][i] * self.f_clusters[k])[0, 0] * (self.model_para['lam_m'] * -0.5)
    #         # t[i, 0] = 0.0
    #         for j in range(size_cluster):
    #             t[i, 0] += np.trace(self.L_clusters[k][i] * self.S_clusters[k][i] * self.L_clusters[k][j])
    #             # t[i, 0] += np.trace(self.L_clusters[k][i] * self.L_clusters[k][j])
    #
    #     # calculate A
    #     for i in range(size_cluster):
    #         # A[i, size_cluster] = -1
    #         # A[size_cluster, i] = 1
    #         # A[i, i] = self.model_para['lam_r']
    #         for j in range(i, size_cluster):
    #             A[i, j] = np.trace(self.L_clusters[k][i] * self.S_clusters[k][i] * self.L_clusters[k][j]) * size_cluster
    #             # A[i, j] = np.trace(self.L_clusters[k][i] * self.L_clusters[k][j]) * size_cluster
    #             A[j, i] = A[i, j]
    #             if i == j:
    #                 A[i, i] -= self.model_para['lam_r']
    #     solved_alpha = solve(A, t)
    #     print 'solved alpha:', solved_alpha
    #     for i in range(size_cluster):
    #         self.alpha_clusters[k][i, 0] = solved_alpha[i, 0]

    def _calc_alpha_cluster(self, k):
        # print 'before optimization, beta: ', self.alpha
        size_cluster = self.alpha_clusters[k].shape[0]
        A = np.matrix(np.zeros((size_cluster + 1, size_cluster + 1)))
        t = np.matrix(np.ones((size_cluster + 1, 1)))

        # calculate t
        for i in range(size_cluster):
            t[i, 0] = (self.f_clusters[k].T * self.L_clusters[k][i] * self.f_clusters[k])[0, 0] * (self.model_para['lam_m'] * -0.5)
            # t[i, 0] = 0.0
            for j in range(size_cluster):
                t[i, 0] += np.trace(self.L_clusters[k][i] * self.S_clusters[k][i] * self.L_clusters[k][j])
                # t[i, 0] += np.trace(self.L_clusters[k][i] * self.L_clusters[k][j])

        # calculate A
        for i in range(size_cluster):
            A[i, size_cluster] = -1
            A[size_cluster, i] = 1
            # A[i, i] = self.model_para['lam_r']
            for j in range(i, size_cluster):
                A[i, j] = np.trace(self.L_clusters[k][i] * self.S_clusters[k][i] * self.L_clusters[k][j]) * size_cluster
                # A[i, j] = np.trace(self.L_clusters[k][i] * self.L_clusters[k][j]) * size_cluster
                A[j, i] = A[i, j]
                if i == j:
                    A[i, i] += self.model_para['lam_r']

        # calculate alpha
        # solved_alpha = A.I * t
        # print 'A:\n', A
        solved_alpha = solve(A, t)
        # print 'solved alpha:', solved_alpha
        for i in range(size_cluster):
            self.alpha_clusters[k][i, 0] = solved_alpha[i, 0]
        # print 'current beta:', self.beta

    def _calc_beta(self):
        # print 'before optimization, beta: ', self.alpha
        A = np.matrix(np.zeros((self.clusters + 1, self.clusters + 1)))
        t = np.matrix(np.ones((self.clusters + 1, 1)))

        # calculate t
        for i in range(self.clusters):
            t[i, 0] = (self.f_clusters[i].T * self.C * self.y)[0, 0] * self.model_para['lam_i']

        # calculate A
        for i in range(self.clusters):
            A[i, self.clusters] = -1
            A[self.clusters, i] = 1
            # A[i, i] = self.model_para['lam_r']
            for j in range(i, self.clusters):
                    A[i, j] = (self.f_clusters[i].T * self.C * self.f_clusters[j])[0, 0] * self.model_para['lam_i']
                    A[j, i] = A[i, j]

        # calculate alpha
        # solved_alpha = A.I * t
        solved_beta = solve(A, t)
        # print 'solved beta:', solved_beta
        for i in range(self.clusters):
            self.beta[i, 0] = solved_beta[i, 0]
        # print 'current beta:', self.beta

    def _calc_Wij(self, k, j):
        if k == j:
            L_common = np.matrix(np.zeros((self.n, self.n), dtype=float), dtype=float)
            for j in range(self.alpha_clusters[k].shape[0]):
                L_common += self.alpha_clusters[k][j, 0] * self.L_clusters[k][j]
            return self.model_para['lam_m'] * L_common + (self.model_para['lam_i'] * (self.beta[k, 0] * self.beta[k, 0])) * self.C
        else:
            return self.model_para['lam_i'] * self.beta[k, 0] * self.beta[j, 0] * self.C

    def _calc_f_close(self):
        L = 0
        for i in range(0, self.clusters):
            temp = self._calc_Wij(i, 0)
            for j in range(1, self.clusters):
                temp = np.column_stack((temp, self._calc_Wij(i, j)))
            if i == 0:
                L = copy.copy(temp)
            else:
                L = np.row_stack((L, temp))
        t = self.C * self.y * (self.model_para['lam_i'] * self.beta[0, 0])
        for i in range(1, self.clusters):
            t = np.row_stack((t, self.C * self.y * (self.model_para['lam_i'] * self.beta[i, 0])))

        solved_f = solve(L, t)
        start_index = 0
        for i in range(0, self.clusters):
            self.f_clusters[i] = solved_f[start_index:start_index + self.n, 0:1]
            start_index += self.n

    def _calc_f(self):
        for i in range(self.clusters):
            b = np.matrix(np.zeros((self.n, 1)))
            for j in range(self.clusters):
                if not j == i:
                    b += self.f_clusters[j] * self.beta[j, 0]
            b = self.C * (self.y - b) * (self.model_para['lam_i'] * self.beta[i, 0])

            L_common = np.matrix(np.zeros((self.n, self.n), dtype=float), dtype=float)
            for j in range(self.alpha_clusters[i].shape[0]):
                L_common += self.alpha_clusters[i][j, 0] * self.L_clusters[i][j]

            self.f_clusters[i] = np.matrix(solve(self.model_para['lam_m'] * L_common +
                                                 (self.model_para['lam_i'] * (self.beta[i, 0] * self.beta[i, 0])) * self.C, b))
            # print i, '-th f updated, loss:', self._loss()

    def _loss(self):
        loss_gr = 0.0
        loss_cs = 0.0
        for i in range(self.clusters):
            L_common = np.matrix(np.zeros((self.n, self.n), dtype=float), dtype=float)
            for j in range(self.alpha_clusters[i].shape[0]):
                L_common += self.alpha_clusters[i][j, 0] * self.L_clusters[i][j]
            loss_gr += ((self.f_clusters[i].T * L_common * self.f_clusters[i]))[0, 0] * (0.5 * self.model_para['lam_m'])
            for j in range(self.alpha_clusters[i].shape[0]):
                loss_cs += np.trace((L_common - self.L_clusters[i][j]) * self.S_clusters[i][j] * (L_common - self.L_clusters[i][j])) * 0.5
                # loss_cs += np.trace((L_common - self.L_clusters[i][j]) * (L_common - self.L_clusters[i][j])) * 0.5

        late_f = np.matrix(np.zeros((self.n, 1)))
        for i in range(self.clusters):
            late_f += self.f_clusters[i] * self.beta[i, 0]
        # print
        loss_hr = (0.5) * self.model_para['lam_i'] * (((late_f - self.y).T * self.C * (late_f - self.y))[0, 0])
        loss_r = 0.0
        for i in range(self.clusters):
            loss_r += (0.5) * self.model_para['lam_r'] * np.power(LA.norm(self.alpha_clusters[i]), 2)
        return [loss_gr + loss_hr + loss_cs + loss_r, loss_hr, loss_gr, loss_cs, loss_r]
        # return [loss_gr + loss_hr + loss_cs, loss_hr, loss_gr, loss_cs]

    def _optimize(self):
        # construct alpha
        alpha_clusters = []
        for i in range(self.clusters):
            alpha = np.empty((self.alpha_clusters[i].shape[0], 1))
            alpha.fill(1.0 / self.alpha_clusters[i].shape[0])
            alpha = np.matrix(alpha)
            alpha_clusters.append(copy.copy(alpha))
        self.alpha_clusters = copy.copy(alpha_clusters)
        # construct beta
        self.beta = np.empty((self.clusters, 1))
        self.beta.fill(1.0 / self.clusters)
        self.beta = np.matrix(self.beta)
        loss = self._loss()
        print 'init loss:', loss
        iter_count = 0
        while True:
            # # self._calc_f()
            self._calc_f_close()
            # cur_loss = self._loss()
            # print 'loss after f:', cur_loss
            self._calc_beta()
            # cur_loss = self._loss()
            # print 'loss after beta:', cur_loss
            self._calc_alpha()
            # cur_loss = self._loss()
            # print 'loss after alpha:', cur_loss
            cur_loss = self._loss()
            if abs(cur_loss[0] - loss[0]) < 0.001:
                print 'converge@', iter_count
                break
            loss = copy.copy(cur_loss)
            if iter_count > 200:
                print 'arrive max iteration'
                break
            iter_count += 1
        print 'final loss:', loss

    def _read_laplacian_matrices_by_cluster(self):
        cluster_fnames = {}
        with open(self.clusters_fnames_fname) as fin:
            cluster_fnames = json.load(fin)
            fin.close()
        print '#clusters:', len(cluster_fnames)
        self.clusters = len(cluster_fnames)
        for clu in cluster_fnames.itervalues():
            if len(clu) < 1:
                print 'unexpected cluster without file'
                exit()
            size_cluster = len(clu)
            # construct alpha
            alpha = np.empty((size_cluster, 1))
            alpha.fill(1.0 / size_cluster)
            alpha = np.matrix(alpha)
            self.alpha_clusters.append(copy.copy(alpha))
            # read in Laplacian matrices one by one and construct the binary S matrix
            Ls = []
            Ss = []
            for fname in clu:
                data = np.genfromtxt('./laplacian_matrices/' + fname, delimiter=',')
                print 'read data with shape:', data.shape, 'from file:', fname
                Ls.append(copy.copy(np.matrix(data)))
                S = np.identity(self.n, dtype=float)
                # sim_data = data - np.identity(self.n, dtype=float)
                for i in range(self.n):
                    all_zero = True
                    for j in range(self.n):
                        if j == i:
                            continue
                        if abs(data[i, j]) > 1e-8:
                            all_zero = False
                            break
                    if all_zero:
                        S[i, i] = 0.0
                Ss.append(copy.copy(np.matrix(S)))
            self.L_clusters.append(copy.copy(Ls))
            self.S_clusters.append(copy.copy(Ss))