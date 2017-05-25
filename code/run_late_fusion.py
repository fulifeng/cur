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


def _gen_grid_search_all_para(parameter_combinations, paras, para_values, selected_paras, para_bounds, step_lengths, para_index):
    if para_index == 0:
        for i in range(len(selected_paras)):
            values = []
            if selected_paras[i] == 'k':
                _gen_rec_search(values, para_bounds[i][0], para_bounds[i][1], para_bounds[i][0], 9)
            else:
                _gen_rec_search(values, para_bounds[i][0], para_bounds[i][1], para_bounds[i][0], 9)
            para_values.append(values)
            # # DEBUG
            # print values
    if para_index == len(selected_paras) - 1:
        for value in para_values[para_index]:
            paras[selected_paras[para_index]] = value
            parameter_combinations.append(paras.copy())
    else:
        for value in para_values[para_index]:
            paras[selected_paras[para_index]] = value
            _gen_grid_search_all_para(parameter_combinations, paras, para_values, selected_paras, para_bounds, step_lengths,
                                           para_index + 1)


def _gen_rec_search(values, low, high, step_length, count):
    # for i in range(9):
    for i in range(count):
        cur_para = low + i * step_length
        if cur_para > high:
            return
        values.append(cur_para)
    _gen_rec_search(values, low * 10, high, step_length * 10, count)


def _get_para_performances(performances):
    aver_mac_f = 0.0
    aver_mic_f = 0.0
    aver_kappa = 0.0
    for i in range(len(performances)):
        aver_mac_f += performances['mac_f']
        aver_mic_f += performances['mic_f']
        aver_kappa += performances['kap']
    return [aver_mac_f / folds, aver_mic_f / folds, aver_kappa / folds]


def grid_search(model, validator, evaluator, model_paras, selected_paras, para_bounds, cross_val=True):
    init_paras = model_paras
    gen_ranking = model.ranking()
    performance = evaluator.evaluate(gen_ranking)
    print 'init parameter:', init_paras
    print 'init performance:', performance
    best_per = [performance['mac_f'], performance['mic_f'], performance['kap']]
    step_lengths = []
    parameter_combinations = []
    para_values = []
    _gen_grid_search_all_para(parameter_combinations, {}, para_values, selected_paras, para_bounds, step_lengths, 0)
    best_para = copy.copy(init_paras)
    for cur_para in parameter_combinations:
        # change parameters
        for para_kv in cur_para.iteritems():
            init_paras[para_kv[0]] = para_kv[1]
        model.model_para = copy.copy(init_paras)
        print model.model_para
        gen_ranking = model.ranking()
        cur_per = []
        if cross_val:
            validator.validate(gen_ranking)
            cur_per = _get_para_performances(validator.dev_pers)
        else:
            performance = evaluator.evaluate(gen_ranking)
            cur_per = [performance['mac_f'], performance['mic_f'], performance['kap']]
        print cur_per
        if cur_per[1] > best_per[1]:
            best_per = copy.copy(cur_per)
            best_para = copy.copy(init_paras)
            print 'better parameters:', best_para, 'with performance:', best_para
    print 'best performance:', best_per
    print 'best parameter:', best_para
    return best_para


def grid_search_cv(model, validator, model_paras, selected_paras, para_bounds):
    init_paras = model_paras
    model.model_para = copy.copy(init_paras)
    validator.validate()
    print 'init parameter:', init_paras
    print 'init performance:'
    best_per_list = []
    best_para_list = []
    for i in range(folds):
        print [validator.dev_pers[i]['mac_f'], validator.dev_pers[i]['mic_f'], validator.dev_pers[i]['kap']]
        best_per_list.append([validator.dev_pers[i]['mac_f'], validator.dev_pers[i]['mic_f'], validator.dev_pers[i]['kap']])
        best_para_list.append(copy.copy(init_paras))
    step_lengths = []
    parameter_combinations = []
    para_values = []
    _gen_grid_search_all_para(parameter_combinations, {}, para_values, selected_paras, para_bounds, step_lengths, 0)
    for cur_para in parameter_combinations:
        # change parameters
        for para_kv in cur_para.iteritems():
            init_paras[para_kv[0]] = para_kv[1]
        model.model_para = copy.copy(init_paras)
        print model.model_para
        # gen_ranking = model.ranking()
        validator.validate()
        for i in range(folds):
            # cur_per = _get_para_performances(validator.dev_pers)
            cur_per = [validator.dev_pers[i]['mac_f'], validator.dev_pers[i]['mic_f'], validator.dev_pers[i]['kap']]
            print 'fold', i, cur_per
            if cur_per[1] > best_per_list[i][1]:
                best_per_list[i] = copy.copy(cur_per)
                best_para_list[i] = copy.copy(cur_para)
                print 'fold:', i, 'better parameters:', best_para_list[i], 'with performance:', best_per_list[i]
    for i in range(folds):
        print 'best performance:', best_per_list[i]
        print 'best parameter:', best_para_list[i]
    return best_para_list


def late_fusion_cv():
    os.chdir(working_dir)
    model_paras = {'lam_i': 20.0, 'thres': 0.002}
    model = late_fusion.late_fusion(model_paras, feature_fname_fname, final_list_fname, historical_ranking_list, hr_scale=2)
    # evaluator = feval.fold_evaluator(gt_fname)
    best_para_list = []
    for i in range(folds):
        # model_paras['thres'] *= 2
        best_para_list.append(copy.copy(model_paras))
    tune = True
    tune = False
    if tune:
        validator = cv.cross_validation(model, gt_fname, folds, test=False, tune=True)
        selected_paras = ['lam_i', 'thres']
        para_bounds = [[1e1, 1e3], [1e-4, 1e-2]]
        best_para_list = grid_search_cv(model, validator, model_paras, selected_paras, para_bounds)

    # gen_ranking = model.ranking()
    # gen_rank_sorted = sorted(gen_ranking.items(), key=operator.itemgetter(1))
    # for rank_kv in gen_rank_sorted:
    #     print rank_kv[0], rank_kv[1]
    print '---------------------------------'
    validator = cv.cross_validation(model, gt_fname, folds, test=True, tune=False)
    validator.testing(best_para_list)
    print '---------------------------------'
    print 'final perfermance:'
    for i in range(folds):
        print 'fold:', i
        per_print(validator.performances[i])
late_fusion_cv()


def late_fusion_full():
    os.chdir(working_dir)
    model_paras = {'lam_i': 300.0, 'thres': 0.001}
    model = late_fusion.late_fusion(model_paras, feature_fname_fname, final_list_fname, historical_ranking_list, hr_scale=2)
    evaluator = feval.fold_evaluator(gt_fname)
    tune = True
    # tune = False
    if tune:
        # validator = cv.cross_validation(model, gt_fname, folds, test=True, tune=False)
        selected_paras = ['lam_i', 'thres']
        para_bounds = [[1e1, 1e3], [1e-3, 1e-2]]
        best_per = grid_search(model, None, evaluator, model_paras, selected_paras, para_bounds, cross_val=False)
        model.model_para = copy.copy(best_per)

    gen_ranking = model.ranking()
    # gen_rank_sorted = sorted(gen_ranking.items(), key=operator.itemgetter(1))
    # for rank_kv in gen_rank_sorted:
    #     print rank_kv[0], rank_kv[1]
    per = evaluator.evaluate(gen_ranking, detail=True)
    print '---------------------------------'
    print 'final perfermance:'
    per_print(per)
# late_fusion_full()

