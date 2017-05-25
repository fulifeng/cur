import argparse
import numpy as np
try:
    set
except NameError:
    from sets import Set as set
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
from util_io import read_ranking_list


class fold_evaluator:
    # def __init__(self, gt_pair_fname, gt_list_fname):
    def __init__(self, gt_pair_fname):
        self.gt_pair_fname = gt_pair_fname
        # self.gt_list_fname = gt_list_fname
        # read in ground truth pairs
        self.gt_pairs = []
        self.gts = 0
        self.read_gt_pairs()
        self.y_true = np.arange(self.gts)
        self.construct_y_true()

    def construct_y_true(self):
        # the y_true is labels of pairs [<A,B>, <B,A>, ...]
        for ind, pair in enumerate(self.gt_pairs):
            label = pair[2]
            self.y_true[ind * 2] = label
            self.y_true[ind  * 2 + 1] = -1 * label

        # check whether the construction works well, if not, there is at least one entry with value not belong to the labels, -1, 0, 1
        for ind, label in enumerate(self.y_true):
            if not (label == -1 or label == 0 or label == 1):
                print 'unexpected value in y_true at index:', ind
                return
        print 'y_true constructed successfully'

    '''
        generated_ranking_list
    '''
    def evaluate(self, generated_ranking, detail=False):
        y_pred = np.arange(self.gts, dtype=int)
        # construct y_pred
        for ind, pair in enumerate(self.gt_pairs):
            if generated_ranking[pair[0]] < generated_ranking[pair[1]]:
                y_pred[ind * 2] = 1
                y_pred[ind * 2 + 1] = -1
            elif generated_ranking[pair[0]] > generated_ranking[pair[1]]:
                y_pred[ind * 2] = -1
                y_pred[ind * 2 + 1] = 1
            else:
                y_pred[ind * 2] = 0
                y_pred[ind * 2 + 1] = 0
        # evaluate
        performance = {}
        mac_p, mac_r, mac_f, _ = precision_recall_fscore_support(self.y_true, y_pred, average='macro')
        performance['mac_p'] = mac_p
        performance['mac_r'] = mac_r
        performance['mac_f'] = mac_f
        mic_p, mic_r, mic_f, _ = precision_recall_fscore_support(self.y_true, y_pred, average='micro')
        performance['mic_p'] = mic_p
        performance['mic_r'] = mic_r
        performance['mic_f'] = mic_f
        performance['kap'] = cohen_kappa_score(self.y_true, y_pred)
        if detail == True:
            print 'confusion matrix:\n', confusion_matrix(self.y_true, y_pred)
        return performance

    def read_gt_pairs(self):
        data = np.genfromtxt(self.gt_pair_fname, delimiter=',', dtype=str)
        print 'ground truth pair read in with shape:', data.shape
        positive_pair = 0
        zero_pair = 0
        for pair in data:
            # positive pairs
            if pair[2] == '1':
                positive_pair += 1
                gt_pair = [pair[0], pair[1], 1]
            # zero pairs
            elif pair[2] == '0':
                zero_pair += 1
                gt_pair = [pair[0], pair[1], 0]
            else:
                print 'unexpected pair:', pair
                break
            self.gt_pairs.append(gt_pair)
        print '#positive pairs:', positive_pair
        print '#zero pairs:', zero_pair
        self.gts = 2 * (positive_pair + zero_pair)
        # number check
        positive_pair = 0
        zero_pair = 0
        for values in self.gt_pairs:
            if values[2] == 1:
                positive_pair += 1
            elif values[2] == 0:
                zero_pair += 1
        print '#positive pairs in the gt pairs:', positive_pair
        print '#zero pairs in the gt pairs:', zero_pair

if __name__ == "__main__":
    desc = "calculate performance of given genenrated ranking list"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-grp', help='name of file with ground truth pairs')
    # parser.add_argument('-grl', help='name of file with ground truth list')
    parser.add_argument('-rl', help='name of file with generated ranking list')
    args =parser.parse_args()

    if args.grp is None:
        args.grp = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/ranking_lists/ground_truth_2016.csv'
    # if args.grl is None:
    #     args.grl = 'ground_truth_list.csv'
    if args.rl is None:
        # args.rl = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/' \
        #           'univ_indicators_benke_12level.csv-official-names.csv-ranking.csv'
        args.rl = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/university_lines_aver_rank_all.csv'
            # read in generated ranking list
        # args.rl = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/ranking_lists/ranking_aver_2015.csv'

    # TEST
    # if args.rl is None:
    #     args.rl = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/test/grl_test.csv'
    # if args.grp is None:
    #     args.grp = '/home/ffl/nus/MM/complementary/chinese_university_ranking/data/features/test/gt_test.csv'

    generated_ranking_list = read_ranking_list(args.rl, dtype=float)
    # data = np.genfromtxt(args.rl, delimiter=',', dtype=str)
    # print 'data read in with shape:', data.shape
    # generated_ranking_list = []
    # for uni in data:
    #     generated_ranking_list.append(data[0])
    # evaluator = evaluator(args.grp, args.grl)
    evaluator = fold_evaluator(args.grp)
    performance = evaluator.evaluate(generated_ranking_list, detail=True)
    print performance