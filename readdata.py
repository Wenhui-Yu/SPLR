#-*-coding:utf-8-*-#

import json
def readdata(dataset):
    #file paths
    '''path_train = 'E:\dataset\\5_score\interactions' + dataset + '_train.json'
    path_train_aux = 'E:\dataset\\5_score\interactions' + dataset + '_train_aux.json'
    path_validate = 'E:\dataset\\5_score\interactions' + dataset + '_validate.json'
    path_test = 'E:\dataset\\5_score\interactions' + dataset + '_test.json' '''
    path_train = 'E:\dataset\interactions' + dataset + '_train.json'
    path_train_aux = 'E:\dataset\interactions' + dataset + '_train_aux.json'
    path_validate = 'E:\dataset\interactions' + dataset + '_validate.json'
    path_test = 'E:\dataset\interactions' + dataset + '_test.json'
    # read files
    with open(path_train) as f:
        line = f.readline()
        train_data = json.loads(line)
    f.close()
    with open(path_train_aux) as f:
        line = f.readline()
        train_data_aux = json.loads(line)
    f.close()
    with open(path_validate) as f:
        line = f.readline()
        validate_data = json.loads(line)
    f.close()
    with open(path_test) as f:
        line = f.readline()
        test_data = json.loads(line)
    f.close()
    num = {'': [39371, 23022],
           '_Women': [35059, 14500],
           '_Men': [22547, 5460],
           '_CLothes': [32728, 8777],
           '_Shoes': [32538, 8231],
           '_Jewelry': [15924, 3607]}
    [P, Q] = num[dataset]
    return train_data, train_data_aux, validate_data, test_data, P, Q