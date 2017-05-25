# -*- coding: utf-8 -*-
import csv
from difflib import SequenceMatcher
import json
import math
import numpy as np
import re
try:
    set
except NameError:
    from sets import Set as set
from sklearn import metrics
import sys
import unicodedata


def read_csv(fname, dtype, skip_header):
        # here the dtype is setted to str as the genfromtxt function will choose a data type for each row by itself.
        print 'read from: ', fname
        return np.genfromtxt(fname, delimiter=',', dtype=dtype, skip_header=skip_header)


def read_shit_csv(fname, skip_header):
        row_index = 0
        data = []
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row_index < skip_header:
                    continue
                else:
                    data.append(row)
                row_index += 1
            f.close()
        return np.matrix(data)


def write_csv_matrix(fname, M, accuracy):
        ofile = open(fname, 'w')
        mode_str = '{:.' + str(accuracy) + 'f}'
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if j == 0:
                    ofile.write(mode_str.format(M[i, j]))
                else:
                    ofile.write(',' + str(mode_str.format(M[i, j])))
            ofile.write('\n')
        ofile.close()


def clean_uni_name(uni_name):
    uni_name = uni_name.replace("（", "")
    uni_name = uni_name.replace('）', '')
    uni_name = uni_name.replace('(', '')
    uni_name = uni_name.replace(')', '')
    if uni_name == '中国矿业大学徐州':
        print 'shit name 中国矿业大学徐州 is changed as:', uni_name
        uni_name = '中国矿业大学'
    if uni_name == '中国矿业大学华东':
        uni_name = '中国矿业大学'
        print 'shit name 中国矿业大学华东 is changed as:', uni_name
    if uni_name == '中国地质大学':
        uni_name = '中国地质大学武汉'
        print 'shit name 中国地质大学 is changed as:', uni_name
    if uni_name == '中国石油大学':
        uni_name = '中国石油大学华东'
        print 'shit name 中国石油大学 is changed as:', uni_name
    if uni_name == '华北电力大学北京':
        uni_name = '华北电力大学'
        print 'shit name 华北电力大学(北京) is changed as:', uni_name
    return uni_name


def longest_common_substring(str1, str2):
    s = SequenceMatcher(None, str1, str2)
    matcher = s.find_longest_match(0, len(str1), 0, len(str2))
    return str1[matcher[0]: matcher[0] + matcher[2]]


def read_ranking_list(fname, dtype=int):
    data = np.genfromtxt(fname, delimiter=',', dtype=str)
    print 'data shape:', data.shape, 'in file:', fname
    ranking_list = {}
    if dtype == int:
        print 'rank data in type: int'
        for uni in data:
            if uni[0] == 'university':
                continue
            ranking_list[uni[0]] = int(uni[1])
    elif dtype == float:
        print 'rank data in type: float'
        for uni in data:
            if uni[0] == 'university':
                continue
            ranking_list[uni[0]] = float(uni[1])
    else:
        print 'unexpected data type'
    return ranking_list


def read_historical_ranking(final_list, hr_fname, scale = 0):
    data = read_ranking_list(hr_fname, dtype=float)
    hr_list = []
    for name in final_list:
        if name in data.keys():
            hr_list.append(data[name])
        else:
            print 'without:', name
            hr_list.append(2000.0)
    # post processing
    amax = -1.0
    for rank in hr_list:
        if abs(rank - 2000) > 1e-8 and rank > amax:
            amax = rank
    for i in range(len(hr_list)):
        if abs(hr_list[i] - 2000) < 1e-8:
            hr_list[i] = amax
    hr_list = np.array(hr_list)
    if scale == 1:
        np.sqrt(hr_list, hr_list)
    elif scale == 2:
        hr_list *= (1.0 / amax)
    return hr_list


'''return kldm_values = {year: {ssdm: [kldm1, kldm2]}}'''
def read_kldm_values(fname):
    kldm_values = {}
    response_values = set()
    print 'read from:', fname
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = json.loads(line)
            year = int(value['year'])
            if not year in kldm_values.keys():
                kldm_values[year] = {}
            province = int(value['ssdm'])
            kls = []
            for item in value['response']:
                kls.append(int(item['key']))
                response_values.add(item['value'])
            kldm_values[year][province] = kls
        f.close()
    for value in response_values:
        print value
    return kldm_values

'''
return kldm_values = {'year': {'ssdm': ['kldm1', 'kldm2']}}
why I write this piece of shit code:
    to ensure the kldm values are in unicode data type so that we can use them as key to search lines in the university_lines and province_lines
'''
def read_kldm_uni_values(fname):
    kldm_values = {}
    response_values = set()
    print 'read from:', fname
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = json.loads(line)
            year = value['year']
            if not year in kldm_values.keys():
                kldm_values[year] = {}
            province = value['ssdm']
            kls = []
            for item in value['response']:
                kls.append(item['key'])
                response_values.add(item['value'])
            kldm_values[year][province] = kls
        f.close()
    for value in response_values:
        print value
    return kldm_values


def handle_locations(fname):
    with open(fname) as f:
        lines = f.readlines()
        f.close()
        values = {}
        for line in lines:
            if len(line) > 10:
                tokens = line.split('"')
                if len(tokens) == 3:
                    value = (int(tokens[1]))
                    location = tokens[2].strip('>').split('<')[0]
                    values[value] = location
                    print 'location: %s, value: %d' % (location, value)
                else:
                    print 'unexpected lines'
        print '#values:', len(values)
        with open(fname + '.json', 'w') as fout:
            json.dump(values, fout, ensure_ascii=False)


'''return kldm_values = {year: {ssdm: [key1: kldm1, key2: kldm2]}}'''
def read_kldm_key_values(fname):
    kldm_key_values = {}
    response_values = set()
    print 'read from:', fname
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = json.loads(line)
            year = int(value['year'])
            if not year in kldm_key_values.keys():
                kldm_key_values[year] = {}
            province = int(value['ssdm'])
            kls = {}
            for item in value['response']:
                kls[item['value']] = int(item['key'])
                response_values.add(item['value'])
            kldm_key_values[year][province] = kls
        f.close()
    for value in response_values:
        print value
    return kldm_key_values


def write_unicode_matrix(table, fname):
    with open(fname, 'w') as f:
        for i, row in enumerate(table):
            if len(row) > 0:
                if i == 0:
                    f.write(row[0].encode('utf-8'))
                else:
                    f.write('\n' + row[0].encode('utf-8'))
            for j in range(1, len(row)):
                f.write(',' + row[j])


def write_matrix_unicode_header(table, fname):
    with open(fname, 'w') as f:
        for i, row in enumerate(table):
            if len(row) > 0:
                if i == 0:
                    f.write(row[0].encode('utf-8'))
                else:
                    f.write('\n' + row[0].encode('utf-8'))
            for j in range(1, len(row)):
                f.write(',' + str(row[j]))


# def remove_punctuation(ori_str):
#     return re.sub(ur"\p{P}+", "", ori_str)
tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                    if unicodedata.category(unichr(i)).startswith('P'))


def remove_punctuation(ori_str):
    return ori_str.translate(tbl)


def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )


def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )


def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
        # return data.decode('utf-8', 'replace')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


def per_print(performance):
    print performance['mac_p']
    print performance['mac_r']
    print performance['mac_f']
    print performance['mic_p']
    print performance['mic_r']
    print performance['mic_f']
    print performance['kap']


def feature_scale(X):
    for j in range(X.shape[1]):
        fmax = np.amax(X[:,j])
        fmin = np.amin(X[:,j])
        # the minimum is missing data
        if abs(fmin + 1) < 1e-8:
            fmin = fmax
            for i in range(X.shape[0]):
                if abs(X[i,j] + 1) > 1e-8 and X[i,j] < fmin:
                    fmin = X[i,j]
        if abs(fmax - fmin) < 1e-8:
            continue
        # scaling
        feature_range = fmax - fmin
        for i in range(X.shape[0]):
            if abs(X[i, j] + 1) > 1e-8:
                X[i, j] = (X[i, j] - fmin) / feature_range
    # return X


def calc_conv_laplacian(X):
    # find data missing lines
    miss_index = []
    for i in range(X.shape[0]):
        if abs(np.amax(X[i]) + 1) < 1e-8:
            miss_index.append(i)
    missings = len(miss_index)
    print '#missing data:', missings
    # calculate median
    euc_dis = metrics.pairwise.euclidean_distances(X, X)
    for ind in miss_index:
        euc_dis[ind] = 0.0
        euc_dis[:, ind] = 0.0
    median = float(np.median(euc_dis))
    # As there are missing data, we cannot use the median function directly, as the real median is in the percentage position:
    #   50 + missing_percentage / 2
    # pos_real_median = 50 + 50.0 * (2.0 * missings * X.shape[0] - missings * missings) / (X.shape[0] * X.shape[0])
    median = np.percentile(euc_dis, 50 + 50 * (2 * missings * X.shape[0] - missings * missings) / (X.shape[0] * X.shape[0]))
    infinity_matrix = metrics.pairwise.rbf_kernel(X, gamma= 0.5 / (median ** 2))
    infinity_matrix -= np.identity(X.shape[0])
    # calculate Laplacian matrix
    # assign 0 to missing lines
    for ind in miss_index:
        infinity_matrix[ind] = 0.0
        infinity_matrix[:, ind] = 0.0
    degree = np.sum(infinity_matrix, axis=0)
    for i in range(len(degree)):
        if abs(degree[i]) > 1e-8:
            degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    d_neg_half_power = np.diag(degree)
    return np.identity(X.shape[0]) - np.dot(np.dot(d_neg_half_power, infinity_matrix), d_neg_half_power)


'''
    The return is in ndarray type
'''
def calc_hyper_laplacian(X, k):
    # find data missing lines
    # miss_index = []
    miss_index = set()
    for i in range(X.shape[0]):
        if abs(np.amax(X[i]) + 1) < 1e-8:
            # miss_index.append(i)
            miss_index.add(i)
    missings = len(miss_index)
    print '#missing data:', missings
    # calculate median
    euc_dis = metrics.pairwise.euclidean_distances(X, X)
    for ind in miss_index:
        euc_dis[ind] = 0.0
        euc_dis[:, ind] = 0.0
    # median = float(np.median(euc_dis))
    # As there are missing data, we cannot use the median function directly, as the real median is in the percentage position:
    #   50 + missing_percentage / 2
    # pos_real_median = 50 + 50.0 * (2.0 * missings * X.shape[0] - missings * missings) / (X.shape[0] * X.shape[0])
    median = np.percentile(euc_dis, 50 + 50 * (2 * missings * X.shape[0] - missings * missings) / (X.shape[0] * X.shape[0]))
    infinity_matrix = metrics.pairwise.rbf_kernel(X, gamma= 0.5 / (median ** 2))
    infinity_matrix -= np.identity(X.shape[0])

    '''
        construct the matrix to calculate the Hypergraph Laplacian
        D_e is edge degree matrix
        D_v is vertex degree matrix
        H is the incidence matrix
        W is edge weight matrix
    '''
    # assign 0 to missing lines
    for ind in miss_index:
        infinity_matrix[ind] = 0.0
        infinity_matrix[:, ind] = 0.0
    # construct H
    H = np.zeros(infinity_matrix.shape, dtype=float)
    for i in range(infinity_matrix.shape[0]):
        if i in miss_index:
            continue
        pos_sort = np.argsort(infinity_matrix[i, :])
        for j in range(k):
            H[pos_sort[infinity_matrix.shape[0] - j - 1], i] = 1.0

    # construct W
    W_neighbor = infinity_matrix * H
    w = np.sum(W_neighbor, axis=0)
    W = np.diag(w)
    # construct D_e
    D_e_inverse = np.identity(infinity_matrix.shape[0], dtype=float) * (1.0 / k)
    for i in miss_index:
        D_e_inverse[i, i] = 0.0
    # construct D_v
    d_v = np.dot(H, w.T)
    # d_v = np.dot(w, H.T)
    for i in range(len(d_v)):
        if abs(d_v[i]) > 1e-8:
            d_v[i] = 1.0 / d_v[i]
    np.sqrt(d_v, d_v)
    D_v_neg_half_power = np.diag(d_v)
    # calculate Laplacian matrix
    # A = np.dot(H, W)
    # A = np.dot(A, D_e_inverse)
    # A = np.dot(A, H.T)
    # A = np.dot(np.dot(D_v_neg_half_power, A), D_v_neg_half_power)
    return np.identity(X.shape[0]) - np.dot(
                                        np.dot(
                                            np.dot(
                                                np.dot(
                                                    np.dot(
                                                        D_v_neg_half_power,
                                                    H),
                                                W),
                                            D_e_inverse),
                                        H.T),
                                    D_v_neg_half_power)

def hsic(X1, X2):
    assert X1.shape[0] == X2.shape[0], "X1 and X2 have different sample count"
    N = X1.shape[0]
    print 'calculate infinity matrix 1'
    # infinity_matrix1 = self._calculate_infinity_matrix(X1, N)
    # calculate median
    euc_dis = metrics.pairwise.euclidean_distances(X1, X1)
    median = float(np.median(euc_dis))
    K = np.matrix(metrics.pairwise.rbf_kernel(X1, gamma= 0.5 / (median ** 2)))

    print 'calculate infinity matrix 2'
    # infinity_matrix2 = self._calculate_infinity_matrix(X2, N)
    euc_dis = metrics.pairwise.euclidean_distances(X2, X2)
    median = float(np.median(euc_dis))
    L = np.matrix(metrics.pairwise.rbf_kernel(X2, gamma=0.5 / (median ** 2)))
    H = np.matrix(np.identity(N)) - 1.0 / N * np.matrix(np.ones((N, N)))
    Kc = H * K * H
    Lc = H * L * H
    criteria = np.multiply(Kc, Lc).sum()
    print criteria, 1.0 / N * criteria
    return 1.0 / N * criteria