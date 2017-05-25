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


def heuristic_cv():
    # # This is for baseline: HR
    # rl = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/ranking_lists/ranking_aver_2015.csv'
    # This is for baseline: NES
    rl = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/university_lines_aver_rank_all.csv'
    model = heuristic.heuristic(rl)
    validator = cv.cross_validation(model, gt_fname, folds)
    validator.validate()
    for i in range(folds):
        print 'fold:', i
        per_print(validator.performances[i])
        print '--------------------------------------'
heuristic_cv()


def heuristic_full():
    eva = feval.fold_evaluator(gt_fname)
    rl = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/ranking_lists/ranking_aver_2015.csv'
    print rl
    model = heuristic.heuristic(rl)
    per_print(eva.evaluate(model.ranking(), detail=True))
    rl = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/university_lines_aver_rank_all.csv'
    print '---------------------------------------'
    print rl
    model = heuristic.heuristic(rl)
    per_print(eva.evaluate(model.ranking(), detail=True))
# heuristic_full()

