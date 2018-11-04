import numpy as np
import json
def read_feature(feature, dataset, Q):
    path_feature = 'E:\\dataset\\features\\' + feature + '_feature.txt'
    path_dict = 'E:\dataset\id2num_dict\id2num_dict' + dataset + '.json'
    with open(path_dict) as f:
        line = f.readline()
        item_i2num_dict = json.loads(line)
    f.close()
    f = open(path_feature, 'r')
    line = eval(f.readline())
    feature = line[1]
    K = len(feature)
    F = np.zeros((Q, K))
    for i in range(0, Q):
        F[i] = feature
    for line in f:
        line = eval(line)
        item_id = line[0]
        feature = line[1]
        try:
            item_num = item_i2num_dict[item_id]
            F[item_num] = feature
        except:
            continue
    return F