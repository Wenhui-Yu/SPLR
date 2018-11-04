#-*-coding:utf-8-*-#

import json
def readdata_time(dataset):
    #file paths
    path_train_record_aux = 'E:\dataset\interactions' + dataset + '_train_record_aux.json'
    path_train_time_aux = 'E:\dataset\interactions' + dataset + '_train_time_aux.json'
    # read files
    with open(path_train_record_aux) as f:
        line = f.readline()
        train_record_aux = json.loads(line)
    f.close()
    with open(path_train_time_aux) as f:
        line = f.readline()
        train_time_aux = json.loads(line)
    f.close()
    return train_record_aux, train_time_aux, 238