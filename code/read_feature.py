import copy
import json
import numpy as np
from util_io import clean_uni_name, feature_scale


class feature_reader:
    '''
        feature_fname_fname is the filename of feature_fname.json
        type (1|2|3|4) denotes the type of initialization, 1 for early fusion, 2 for late fusion, 3 for our proposed model
        4 for MvDA outputs
    '''
    def __init__(self, feature_fname_fname, final_list_fname, read_type=3):
        self.read_type = read_type
        self.feature_fname_fname = feature_fname_fname
        self.final_list_fname = final_list_fname
        self.final_list = np.genfromtxt(final_list_fname, delimiter=',', dtype=str)
        print 'read a matrix with shape:', self.final_list.shape, 'from:', final_list_fname
        self.feature_fnames = {}
        with open(self.feature_fname_fname, 'r') as fin:
            self.feature_fnames = json.load(fin)

    def read_feature(self):
        if self.read_type == 1:
            return self._read_early_fusion_feature()
        if self.read_type == 2:
            return self._read_late_fusion_feature()

    def _read_early_fusion_feature(self):
        concatenated_feature = []
        is_first = True
        # # TEST
        # start_index = 0
        for perspective in self.feature_fnames.itervalues():
            for types in perspective.itervalues():
                for fname in types:
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
        return concatenated_feature

    def _read_late_fusion_feature(self):
        feature_matrices = []
        feature_names = []
        # # TEST
        # start_index = 0
        for perspective in self.feature_fnames.iteritems():
            concatenated_feature = []
            feature_names.append(perspective[0])
            print 'reading perspective:', perspective[0]
            is_first = True
            for types in perspective[1].itervalues():
                for fname in types:
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
            feature_matrices.append(concatenated_feature)
        return feature_names, feature_matrices

    def transfer_feature(self):
        for perspective in self.feature_fnames.itervalues():
            for types in perspective.itervalues():
                for fname in types:
                    print 'transferring feature from:', fname
                    data = np.genfromtxt(fname, delimiter=',', dtype=str)
                    print 'read data with shape:', data.shape
                    feature_matrix = []
                    missing_line = []
                    for i in range(1, data.shape[1]):
                        missing_line.append(-1)
                    indexes = {}
                    for ind, line in enumerate(data):
                        if line[0] == 'university':
                            continue
                        indexes[clean_uni_name(line[0])] = ind
                    print '#universities:', len(indexes)
                    without_feature = 0
                    for name in self.final_list:
                        if name not in indexes.keys():
                            print 'without:', name
                            feature_matrix.append(missing_line)
                            without_feature += 1
                        else:
                            ind = indexes[name]
                            feature_matrix.append(data[ind][1:])
                    print '#universities in the matrix:', len(feature_matrix)
                    print '#universities without feature:', without_feature
                    ofname = '../matrix_feature/' + fname
                    np.savetxt(ofname, feature_matrix, delimiter=',', fmt='%s')
                    print '-------------------------------------------'
                    print '-------------------------------------------'
                    print '-------------------------------------------'
                    # exit()