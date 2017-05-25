import copy
import cross_validation as cv
import early_fusion
import fold_evaluator as feval
import heuristic
import json
import late_fusion
import mcsir
import multi_modal
from nmf import get_missing_entry
import numpy as np
from numpy import linalg as LA
import operator
import os
import random
import read_feature as rf
try:
    set
except NameError:
    from sets import Set as set
from sklearn import cluster
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, confusion_matrix, accuracy_score
import subspace
import tune_hypergraph
from util_io import read_ranking_list, clean_uni_name, per_print, feature_scale, calc_conv_laplacian, calc_hyper_laplacian, hsic


basic_dir = '/home/ffl/nus/MM/complementary/chinese_university_ranking/code/code_release/cur/data'
gt_fname = os.path.join(basic_dir, 'gt', 'ground_truth_2016_final.csv')
folds = 5
final_list_fname = os.path.join(basic_dir, 'ranking_lists', 'final_list_aver_2016.csv')
feature_fname_fname = os.path.join(basic_dir, 'matrix_feature', 'feature_fnames.json')
historical_ranking_list = os.path.join(basic_dir, 'ranking_lists', 'ranking_aver_2015.csv')
confidence_histo_rank_list = os.path.join(basic_dir, 'ranking_lists', 'confidence_ranking_aver_2015.csv')
working_dir = os.path.join(basic_dir, 'matrix_feature')


def graph_clustering():
    print '-------------Graph Clustering------------'
    os.chdir(working_dir)
    lapla_fnames_fname = './laplacian_matrices/fnames.txt'
    fnames = []
    with open(lapla_fnames_fname) as fin:
        fnames = fin.read().splitlines()
        fin.close()
    print '#laplacian matrices selected'
    # read laplacian matrices
    laplacian_matrix = []
    for fname in fnames:
        data = np.genfromtxt('laplacian_matrices/' + fname, delimiter=',')
        laplacian_matrix.append(data)

    # calcualte similarity matrix
    similarity_matrix = np.ones((len(fnames), len(fnames)), dtype=float)
    for i in range(len(fnames)):
        for j in range(i + 1, len(fnames)):
            similarity_matrix[i, j] = hsic(laplacian_matrix[i], laplacian_matrix[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    print 'similarity matrix:', similarity_matrix

    # clustering
    for i in range(1, similarity_matrix.shape[0]):
        clustering = cluster.SpectralClustering(n_clusters=i, eigen_solver='arpack', affinity='precomputed')
        clustering.fit(similarity_matrix)
        clusters = clustering.fit_predict(similarity_matrix)
        print '#clusters:', i, 'clusters:', clusters
        cluster_dic = {}
        for j in range(len(clusters)):
            key = str(clusters[j])
            if key not in cluster_dic.keys():
                cluster_dic[key] = []
            cluster_dic[key].append(fnames[j])
        with open('laplacian_matrices/clusters_' + str(i) + '.csv', 'w') as fout:
            json.dump(cluster_dic, fout)
            fout.close()
    cluster_dic = {}
    for j in range(len(fnames)):
        key = str(j)
        if key not in cluster_dic.keys():
            cluster_dic[key] = []
        cluster_dic[key].append(fnames[j])
    with open('laplacian_matrices/clusters_' + str(len(fnames)) + '.csv', 'w') as fout:
        json.dump(cluster_dic, fout)
        fout.close()
# graph_clustering()


def gt_compare(gt_fname1, gt_fname2):
    # as one evaluator only stores one ground truth, we declare two evaluators
    e1 = evaluator(gt_fname1)
    e2 = evaluator(gt_fname2)
    n = len(e1.common_names)
    y_pred = np.arange(n * (n - 1), dtype=int)
    ind = 0
    for name1 in e1.common_names:
        for name2 in e1.common_names:
            if name1 == name2:
                continue
            possitive = False
            negative = False
            if name1 in e2.gt_pairs_positive.keys():
                if name2 in e2.gt_pairs_positive[name1].keys():
                    y_pred[ind] = 1
                    ind += 1
                    possitive = True
            if name2 in e2.gt_pairs_positive.keys():
                if name1 in e2.gt_pairs_positive[name2].keys():
                    y_pred[ind] = -1
                    ind += 1
                    negative = True
            if not (possitive == True or negative == True):
                y_pred[ind] = 0
                ind += 1
    # check if the y_pred is constructed successfully
    for ind, label in enumerate(y_pred):
        if not (label == -1 or label == 0 or label == 1):
            print 'unexpected value in y_true at index:', ind
            return
    print 'y_pred constructed successfully'
    print 'macro averaged:', precision_recall_fscore_support(e1.y_true, y_pred, average='macro')
    print 'micro averaged:', precision_recall_fscore_support(e1.y_true, y_pred, average='micro')
    print 'cohen\'s kappa:', cohen_kappa_score(e1.y_true, y_pred)
    print 'accuracy score:', accuracy_score(e1.y_true, y_pred)
    print 'confusion matrix:\n', confusion_matrix(e1.y_true, y_pred)

# gt_compare('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/ranking_lists/ground_truth_2016.csv',
#            '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/ranking_lists/ground_truth_2015.csv')


def line_aver_rank_transfer(rank_fname, indicator_fname):
    origin_ranking = read_ranking_list(rank_fname, dtype=float)
    print 'ranking size:', len(origin_ranking)
    data = read_ranking_list(indicator_fname, dtype=int)
    uni_names = set()
    for uni in data.iteritems():
        if uni[1] == 1:
            uni_names.add(clean_uni_name(uni[0]))
            # uni_names.add(clean_uni_name(uni[0]))
    print '#universities selected:', len(uni_names)
    init_ranking = {}
    for rank_kv in origin_ranking.iteritems():
        init_ranking[clean_uni_name(rank_kv[0])] = rank_kv[1]
    print '#initial ranking:', len(init_ranking)
    # for rank_kv in init_ranking.iteritems():
    #     print rank_kv[0], rank_kv[0]
    for name in uni_names:
        if name not in init_ranking.keys():
            init_ranking[name] = 2000
    init_ranking_sorted = sorted(init_ranking.items(), key=operator.itemgetter(1))
    np.savetxt(rank_fname + '_temp', init_ranking_sorted, fmt='%s', delimiter=',')
# line_aver_rank_transfer('/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/university_lines_aver_rank.csv',
#                         '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/univ_indicators_benke_12level.csv')


def nmf_data_completion():
    data_fname = '/home/ffl/nus/MM/complementary/chinese_university_ranking/experiment/matrix_feature/university_lines_aver_trans.csv'
    # data_fname = '/home/ffl/nus/MM/complementary/chinese_university_ranking/experiment/matrix_feature/feature_ipin.csv'
    data = np.genfromtxt(data_fname, delimiter=',', dtype=float)
    feature_scale(data)
    R = copy.copy(data)
    # selected_test = []
    # tests = R.shape[0] * R.shape[1] / 20
    # print '#tests:', tests
    # for ind in range(tests):
    #     i = random.randint(0, R.shape[0] - 1)
    #     j = random.randint(0, R.shape[1] - 1)
    #     if abs(R[i, j] + 1) > 1e-8:
    #         selected_test.append([i, j, R[i, j]])
    #         R[i, j] = -1
    k = data.shape[1]
    iters = 500
    alpha = 0.0004
    print 'latent concepts:', k, 'iter:', iters, 'alpha:', alpha
    R = get_missing_entry(R, k=k, steps=iters, beta=0.02, missing_denotation=-1)
    np.savetxt(data_fname.replace('.csv', '_nmf.csv'), R, delimiter=',')
# nmf_data_completion()


def refine_ground_truth():
    final_list = np.genfromtxt(final_list_fname, delimiter=',', dtype=str)
    print 'final list shape:', final_list.shape
    gt_fname = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/ranking_lists/ground_truth_2015.csv'
    data = np.genfromtxt(gt_fname, delimiter=',', dtype=str)
    print 'ground truth pair read in with shape:', data.shape
    final_names = set()
    for name in final_list:
        final_names.add(name)
    selected_pair = []
    pos_pair = 0
    zero_pair = 0
    for pair in data:
        # update common names
        if pair[0] in final_names and pair[1] in final_names:
            selected_pair.append(pair)
            if pair[2] == '1':
                pos_pair += 1
            elif pair[2] == '0':
                zero_pair += 1
            else:
                print 'shit'
    print 'selected pos:', pos_pair
    print 'selected zero:', zero_pair
    # write
    ofname = gt_fname.replace('.csv', '_final.csv')
    np.savetxt(ofname, selected_pair, fmt='%s', delimiter=',')
# refine_ground_truth()


def transfer_feature_matrix():
    final_list_fname = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/ranking_lists/final_list_aver_2016.csv'
    feature_fname_fname = '/home/ffl/nus/MM/complementary/chinese_university_ranking/experiment/ori_feature/feature_fnames.json'
    reader = rf.feature_reader(feature_fname_fname, final_list_fname, read_type=5)
    reader.transfer_feature()
# transfer_feature_matrix()


def transfer_university_line_aver():
    fname = '/home/ffl/nus/MM/complementary/chinese_university_ranking/experiment/ori_feature/university_lines_aver.csv'
    data = np.genfromtxt(fname, delimiter=',', dtype=str)
    for i in range(1, data.shape[0]):
        for j in range(1, data.shape[1]):
            d = float(data[i, j])
            if abs(d + 1) < 1e-8:
                print data[i, j]
            if data[i, j] == '-1.0':
                data[i, j] = '-0.999'
            if data[i, j] == '-10000':
                data[i, j] = '-1'
    np.savetxt(fname.replace('.csv', '_trans.csv'), data, delimiter=',', fmt='%s')
# transfer_university_line_aver()


def tune_hyper_k():
    print '--------Tune K for Hypergraph----------'
    os.chdir(working_dir)
    channel = 'official'
    channel = 'mass_media'
    channel = 'general_user'
    # initial performance
    model_paras = {'lam_i': 300.0, 'thres': 0.001}
    model = tune_hypergraph.tune_hypergraph(model_paras, feature_fname_fname, final_list_fname, historical_ranking_list, hr_scale=2,
                                            hyper_k=3, channel=channel)
    gen_ranking = model.ranking()
    evaluator = feval.fold_evaluator(gt_fname)
    performance = evaluator.evaluate(gen_ranking)
    print 'init parameter:', model_paras
    print 'init performance:', performance
    best_per = [performance['mac_f'], performance['mic_f'], performance['kap']]
    best_para = copy.copy(model_paras)
    best_k = 3
    for hyper_k in range(3, 11):
        model_paras = {'lam_i': 300.0, 'thres': 0.001}
        # __init__(self, model_para, feature_fnames_fname, final_list_fname, historical_ranking_list, hr_scale, hyper_k, channel):
        model = tune_hypergraph.tune_hypergraph(model_paras, feature_fname_fname, final_list_fname, historical_ranking_list, hr_scale=2,
                                                hyper_k=hyper_k, channel=channel)
        tune = True
        # tune = False
        if tune:
            # validator = cv.cross_validation(model, gt_fname, folds, test=True, tune=False)
            selected_paras = ['lam_i', 'thres']
            para_bounds = [[1e1, 1e3], [1e-4, 1e-2]]
            # para_bounds = [[1e1, 1e2], [1e-4, 1e-3]]
            cur_best_para = grid_search(model, None, evaluator, model_paras, selected_paras, para_bounds, cross_val=False)
            model.model_para = copy.copy(cur_best_para)
        gen_ranking = model.ranking()
        # gen_rank_sorted = sorted(gen_ranking.items(), key=operator.itemgetter(1))
        # for rank_kv in gen_rank_sorted:
        #     print rank_kv[0], rank_kv[1]
        performance = evaluator.evaluate(gen_ranking, detail=True)
        cur_per = [performance['mac_f'], performance['mic_f'], performance['kap']]
        print cur_per
        # update best performance and best parameters
        if cur_per[1] > best_per[1]:
            best_per = copy.copy(cur_per)
            if tune:
                best_para = copy.copy(cur_best_para)
            else:
                best_para = copy.copy(model_paras)
            best_k = hyper_k
            print 'better parameters:', best_para, '@k:', hyper_k, 'with performance:', best_per
    print 'best performance:', best_per
    print 'best k:', best_k
    print 'best parameter:', best_para
    # get final performance:
    model = tune_hypergraph.tune_hypergraph(best_para, feature_fname_fname, final_list_fname, historical_ranking_list, hr_scale=2,
                                            hyper_k=best_k, channel=channel)
    gen_ranking = model.ranking()
    per = evaluator.evaluate(gen_ranking, detail=True)
    print '---------------------------------'
    print 'final perfermance for k:', best_k
    per_print(per)
# tune_hyper_k()


def write_cc_laplacian_matrices():
    print '-------------Write Laplacian Matrices------------'
    os.chdir(working_dir)
    laplacian_matrices = []
    laplacian_names = []
    feature_fnames = {}
    with open(feature_fname_fname, 'r') as fin:
        feature_fnames = json.load(fin)
        fin.close()
    # construct conventional graphs
    print '-----------Construct Conventional Graphs Like Late Fusion----------'
    for perspective in feature_fnames.iteritems():
        concatenated_feature = []
        laplacian_names.append(perspective[0])
        print 'reading perspective:', perspective[0]
        is_first = True
        for types in perspective[1].iteritems():
            for fname in types[1]:
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
        laplacian_matrices.append(calc_conv_laplacian(concatenated_feature))

    # write_to_files
    assert len(laplacian_matrices) == len(laplacian_names), '#names should be equal to #matrices'
    for i in range(len(laplacian_names)):
        print laplacian_names[i] + '.csv'
        np.savetxt('laplacian_matrices/' + laplacian_names[i] + '-cc.csv', laplacian_matrices[i], delimiter=',')
# write_cc_laplacian_matrices()


def write_laplacian_matrices():
    print '-------------Write Laplacian Matrices------------'
    os.chdir(working_dir)
    laplacian_matrices = []
    laplacian_names = []
    feature_fnames = {}
    with open(feature_fname_fname, 'r') as fin:
        feature_fnames = json.load(fin)
        fin.close()
    # construct conventional graphs
    print '-----------Construct Conventional Graphs----------'
    for perspective in feature_fnames.iteritems():
        concatenated_feature = []
        laplacian_names.append(perspective[0])
        print 'reading perspective:', perspective[0]
        is_first = True
        for types in perspective[1].iteritems():
            if types[0] == 'text':
                continue
            for fname in types[1]:
                print 'reading feature from:', fname
                data = np.genfromtxt(fname, delimiter=',', dtype=float)
                print 'read data with shape:', data.shape
                if is_first:
                    is_first = False
                    concatenated_feature = copy.copy(data)
                else:
                    concatenated_feature = np.concatenate((concatenated_feature, data), axis=1)
                # # TEST
                # print data[0]
                # print concatenated_feature[0][start_index: start_index + data.shape[1]]
                # if not np.array_equal(data[0], concatenated_feature[0][start_index: start_index + data.shape[1]]):
                #     print '!!!Not Equal!!!'
                # start_index += data.shape[1]
                print 'current feature with shape:', concatenated_feature.shape
                print '-----------------------------------------------'
        feature_scale(concatenated_feature)
        laplacian_matrices.append(calc_conv_laplacian(concatenated_feature))

    # construct hypergraphs
    print '-----------Construct Hypergraphs--------------'
    hyper_ks = {'official': 5, 'mass_media': 5, 'general_user': 5}
    for perspective in feature_fnames.iteritems():
        concatenated_feature = []
        print 'reading perspective:', perspective[0]
        is_first = True
        for types in perspective[1].iteritems():
            if types[0] == 'text':
                for fname in types[1]:
                    print 'reading feature from:', fname
                    data = np.genfromtxt(fname, delimiter=',', dtype=float)
                    print 'read data with shape:', data.shape
                    if is_first:
                        is_first = False
                        concatenated_feature = copy.copy(data)
                    else:
                        concatenated_feature = np.concatenate((concatenated_feature, data), axis=1)
                    # # TEST
                    # print data[0]
                    # print concatenated_feature[0][start_index: start_index + data.shape[1]]
                    # if not np.array_equal(data[0], concatenated_feature[0][start_index: start_index + data.shape[1]]):
                    #     print '!!!Not Equal!!!'
                    # start_index += data.shape[1]
                    print 'current feature with shape:', concatenated_feature.shape
                    print '-----------------------------------------------'
        if not is_first:
            laplacian_names.append(perspective[0] + '_hyper')
            feature_scale(concatenated_feature)
            laplacian_matrices.append(calc_hyper_laplacian(concatenated_feature, hyper_ks[perspective[0]]))

    # write_to_files
    assert len(laplacian_matrices) == len(laplacian_names), '#names should be equal to #matrices'
    for i in range(len(laplacian_names)):
        print laplacian_names[i] + '.csv'
        np.savetxt('laplacian_matrices/' + laplacian_names[i] + '.csv', laplacian_matrices[i], delimiter=',')
# write_laplacian_matrices()


def write_for_sl():
    os.chdir(working_dir)
    model_paras = {'lam_i': 300.0, 'thres': 0.001}
    model = subspace.subspace(model_paras, feature_fname_fname, final_list_fname, historical_ranking_list, hr_scale=2)
    model.write_for_matlab()
# write_for_sl()
