#Extension with CNN feature and spectral feature
#Wenhui Yu, 2018.10.29

import numpy as np
from readdata import readdata
from evaluation_F1 import evaluation_F1
from evaluation_NDCG import evaluation_NDCG
from save_result import save_result
from read_feature import read_feature
from numpy import *
import xlwt
import time
import json

## parameters setting
dataset = 5                         # datasets, 3 for Clothing and 5 for Jewelry
eta = 0.05                           # learning rate
eta_1 = 0.01                        # weighting parameter 1
eta_2 = 0.01                        # weighting parameter 2
K0 = 200                            # length of latent factors
K1 = 1000                           # length of user spectral feature
K2 = 0                            # length of item spectral feature
top_k = [2, 5, 10, 20, 50]          # set top k
batch_size_train = 5000             # batch size for training
batch_size_test = 1000              # batch size for test
lambda_r = 0.7                      # regularization coefficient
vali_test = 0                       # select validate set or test set, 0 for validate, 1 for test
feat = [0]
epoch = 200                         # number of epoch

def d(x):
    # delta function for BPR
    if x > 10:
        return 0
    if x < -10:
        return 1
    if x >= -10 and x <= 10:
        return 1.0 / (1.0 + exp(x))

def get_feature(dataset):
    data_path = 'E:\par_mat' + dataset_list[dataset] + '.json'
    with open(data_path) as f:
        line = f.readline()
        train_data = json.loads(line)
    f.close()
    [E, F, R_fou, Lam_u, Lam_v] = train_data
    return np.array(E), np.array(F)

def get_feature_cnn(dataset):
    feat_list = ['CNN', 'AES', 'CH']
    Fc = read_feature(feat_list[feat[0]], dataset, Q)
    for i in range(1, len(feat)):
        Fc = np.hstack((Fc, read_feature(feat_list[feat[i]], feature_length, dataset, Q)))
    return Fc

def test_SPLR(U, V, M, E, N, F, Mc, Fc):
    # test the effectiveness of the model
    U = mat(U)
    V = mat(V)
    M = mat(M)
    F = mat(F)
    N = mat(N)
    E = mat(E)
    Mc = mat(Mc)
    Fc = mat(Fc)
    k_num = len(top_k)
    # k_num-length lists for F1 and NDCG
    F1 = np.zeros(k_num)
    NDCG = np.zeros(k_num)
    num_item = len(Test)
    # select batch_size_test test samples randomly
    for i in range(batch_size_test):
        j = int(math.floor(num_item * random.random()))
        # the form of test data is [u, [i, i, i, i], [r, r, r]]
        u = Test[j][0]
        # test sample :u & [i, i, i, i]
        positive_item = Test[j][1]
        # score all items
        score = U[u] * V.T + E[u] * M.T + N[u] * F.T + Mc[u] * Fc.T   #get the score of each item
        score = score.tolist()[0]  #mat -> list
        # ordering
        b = zip(score, range(len(score)))
        b.sort(key=lambda x: x[0])
        order = [x[1] for x in b]
        order.reverse()
        # remove the positive samples from the item order list
        train_positive = train_data_aux[u][0]
        for item in train_positive:
            order.remove(item)
        # remove the positive samples from the test items
        positive_item = list(set(positive_item) - set(train_positive))
        # test for every top_k
        for i in range(len(top_k)):
            F1[i] += evaluation_F1(order, top_k[i], positive_item)
            NDCG[i] += evaluation_NDCG(order, top_k[i], positive_item)
    # calculate the average
    F1 = (F1 / batch_size_test).tolist()
    NDCG = (NDCG / batch_size_test).tolist()
    return F1, NDCG

def train_SPLR(eta):
    # training
    # matrices initialization
    U = np.array([np.array([(random.random() / math.sqrt(K0)) for j in range(K0)]) for i in range(P)])
    V = np.array([np.array([(random.random() / math.sqrt(K0)) for j in range(K0)]) for i in range(Q)])
    M = np.array([np.array([(random.random() / math.sqrt(K1)) for j in range(K1)]) for i in range(Q)])
    N = np.array([np.array([(random.random() / math.sqrt(K2)) for j in range(K2)]) for i in range(P)])
    Mc = np.array([np.array([(random.random() / math.sqrt(Kc)) for j in range(Kc)]) for i in range(P)])
    e = 100000000000000000000000
    # output the F1 and NDCG before training
    print 'iteration ', 0,
    [F1, NDCG] = test_SPLR(U, V, M, E, N, F, Mc, Fc)
    Fmax = 0
    if F1[0] > Fmax:
        Fmax = F1[0]
    print Fmax, 'F1: ', F1, '  ', 'NDCG1: ', NDCG
    ## save the results in Excel file
    save_result([' '], [''] * len(top_k), [''] * len(top_k), path_excel)
    save_result('metric', ['F1'] * len(top_k), ['NDCG'] * len(top_k), path_excel)
    save_result('Top_k', top_k, top_k, path_excel)
    save_result([' '], [''] * len(top_k), [''] * len(top_k), path_excel)
    save_result('iteration ' + str(0), F1, NDCG, path_excel)

    # get the number of training samples
    Re = len(train_data)
    # split the training data by batch_size_train
    bs = range(0, Re, batch_size_train)
    bs.append(Re)
    # begin iterating
    for ep in range(0, epoch):
        print 'iteration ', ep + 1,
        eta = eta * 0.99
        # enumerate all positive samples
        for i in range(0, len(bs) - 1):
            if abs(U.sum()) < e:
                # for all positive samples, input them as batches
                # initialize dU, dV, dM, dN
                dU = np.zeros((P, K0))
                dV = np.zeros((Q, K0))
                dM = np.zeros((Q, K1))
                dN = np.zeros((P, K2))
                dMc = np.zeros((P, Kc))
                for re in range(bs[i], bs[i + 1]):
                    # [u, i, r]
                    p = train_data[re][0]
                    qi = train_data[re][1]
                    xi = np.dot(U[p], V[qi]) + np.dot(E[p], M[qi]) + np.dot(N[p], F[qi]) + np.dot(Mc[p], Fc[qi])
                    num = 0
                    # select 5 negative samples randomly to calculate the gradient
                    while num < 5:
                        qj = int(random.uniform(0, Q))
                        if not (qj in train_data_aux[p][0]):
                            num += 1
                            xj = np.dot(U[p], V[qj]) + np.dot(E[p], M[qj]) + np.dot(N[p], F[qj]) + np.dot(Mc[p], Fc[qj])
                            xij = xi - xj
                            dU[p] += d(xij) * (V[qi] - V[qj])
                            dV[qi] += d(xij) * U[p]
                            dV[qj] -= d(xij) * U[p]
                            dM[qi] += d(xij) * E[p]
                            dM[qj] -= d(xij) * E[p]
                            dN[p] += d(xij) * (F[qi] - F[qj])
                            dMc[p] += d(xij) * (Fc[qi] - Fc[qj])

                    neighbor = set()
                    neighbor = neighbor | set(graph_cluster_dic[graph_cluster_list[qi]][0])
                    neighbor = neighbor | set(CNN_cluster_dic[CNN_cluster_list[qi]][0])
                    neighbor = list(neighbor)

                    if len(neighbor) > 5:
                        num = 0
                        while num < 5:
                            qj = neighbor[int(random.uniform(0, len(neighbor) - 0.1))]
                            if not (qj in train_data_aux[p][0]):
                                num += 1
                                xj = np.dot(U[p], V[qj]) + np.dot(E[p], M[qj]) + np.dot(N[p], F[qj]) + np.dot(Mc[p], Fc[qj])
                                xij = xi - xj
                                dU[p] += eta_1 * d(xij) * (V[qi] - V[qj])
                                dV[qi] += eta_1 * d(xij) * U[p]
                                dV[qj] -= eta_1 * d(xij) * U[p]
                                dM[qi] += eta_1 * d(xij) * E[p]
                                dM[qj] -= eta_1 * d(xij) * E[p]
                                dN[p] += eta_1 * d(xij) * (F[qi] - F[qj])
                                dMc[p] += eta_1 * d(xij) * (Fc[qi] - Fc[qj])

                        num = 0
                        while num < 5:
                            qj = int(random.uniform(0, Q))
                            if not (qj in train_data_aux[p][0] or qj in neighbor):
                                qi = neighbor[int(random.uniform(0, len(neighbor) - 0.1))]
                                xi = np.dot(U[p], V[qi]) + np.dot(E[p], M[qi]) + np.dot(N[p], F[qi]) + np.dot(Mc[p], Fc[qi])
                                num += 1
                                xj = np.dot(U[p], V[qj]) + np.dot(E[p], M[qj]) + np.dot(N[p], F[qj]) + np.dot(Mc[p], Fc[qj])
                                xij = xi - xj
                                dU[p] += eta_2 * d(xij) * (V[qi] - V[qj])
                                dV[qi] += eta_2 * d(xij) * U[p]
                                dV[qj] -= eta_2 * d(xij) * U[p]
                                dM[qi] += eta_2 * d(xij) * E[p]
                                dM[qj] -= eta_2 * d(xij) * E[p]
                                dN[p] += eta_2 * d(xij) * (F[qi] - F[qj])
                                dMc[p] += eta_2 * d(xij) * (Fc[qi] - Fc[qj])

                # update matrices
                U += eta * (dU - lambda_r * U)
                V += eta * (dV - lambda_r * V)
                M += eta * (dM - lambda_r * M)
                N += eta * (dN - lambda_r * N)
                Mc += eta * (dMc - lambda_r * Mc)
            # test the model after iterating all training data, and save the result
        if abs(U.sum()) < e:
            [F1, NDCG] = test_SPLR(U, V, M, E, N, F, Mc, Fc)
            if F1[0] > Fmax:
                Fmax = F1[0]
            print Fmax, 'F1: ', F1, '  ', 'NDCG1: ', NDCG
            save_result('iteration ' + str(ep + 1), F1, NDCG, path_excel)
        else:
            break

    if abs(U.sum()) < e:
        return 0
    else:
        return 1

def save_parameter():
    # save parameters into excel file
    dataset_list = ['all', '_Women', '_Men', '_CLothes', '_Shoes', '_Jewelry']
    excel = xlwt.Workbook()
    table = excel.add_sheet('A Test Sheet')
    table.write(0, 0, 'model')
    table.write(0, 2, 'extension_SF_CNN')
    table.write(1, 0, 'dataset')
    table.write(1, 2, dataset_list[dataset])
    table.write(2, 0, 'eta')
    table.write(2, 2, eta)
    table.write(2, 4, 'eta1')
    table.write(2, 6, eta_1)
    table.write(2, 8, 'eta2')
    table.write(2, 10, eta_2)
    table.write(3, 0, 'K0')
    table.write(3, 2, K0)
    table.write(3, 4, 'K1')
    table.write(3, 6, K1)
    table.write(3, 8, 'K2')
    table.write(3, 10, K2)
    table.write(4, 0, 'top_k')
    for i in range(len(top_k)):
        table.write(4, 2 + i, top_k[i])
    table.write(5, 0, 'batch_size_train')
    table.write(5, 2, batch_size_train)
    table.write(6, 0, 'batch_size_test')
    table.write(6, 2, batch_size_test)
    table.write(7, 0, 'lambda_r')
    table.write(7, 2, lambda_r)
    table.write(8, 0, 'vali_test')
    table.write(8, 2, vali_test)
    table.write(9, 0, 'epoch')
    table.write(9, 2, epoch)
    table.write(17, 0, ' ')

    excel.save(path_excel)

def print_parameter():
    # print parameters
    print 'model', 'extension_SF_CNN'
    print 'dataset', dataset
    print 'eta', eta, '    eta1', eta_1, '    eta2', eta_2
    print 'K0', K0, '    K1', K1, '    K2', K2
    print 'top_k', top_k
    print 'batch_size_train', batch_size_train
    print 'batch_size_test', batch_size_test
    print 'lambda_r:', lambda_r
    print 'vali_test', vali_test
    print 'epoch', epoch
    print


'''*************************main****************************'''
'''*************************main****************************'''
for i in range(1):
    # dataset list
    dataset_list = ['', '_Women', '_Men', '_CLothes', '_Shoes', '_Jewelry']
    # load data
    [train_data, train_data_aux, validate_data, test_data, P, Q] = readdata(dataset_list[dataset])
    # load feature
    [E, F] = get_feature(dataset)
    '''for i in range(len(E)):
        E[i] = E[i] / math.sqrt(mat(E[i]) * mat(E[i]).T)
    for i in range(len(F)):
        F[i] = F[i] / math.sqrt(mat(F[i]) * mat(F[i]).T)'''
    E = E[:, 0: K1]
    F = F[:, 0: K2]

    Fc = get_feature_cnn(dataset_list[dataset])
    Kc = len(Fc[0])

    # select validate or test dataset
    if vali_test == 0:
        Test = validate_data
    else:
        Test = test_data

    # load clusters
    graph_cluster_path = 'E:\dataset\cluster\graph_cluster' + dataset_list[dataset] + '.json'
    CNN_cluster_path = 'E:\dataset\cluster\CNN_cluster' + dataset_list[dataset] + '.json'
    f = open(graph_cluster_path, 'r')
    line = f.readline()
    [graph_cluster_list, graph_cluster_dic] = json.loads(line)
    f.close()

    f = open(CNN_cluster_path, 'r')
    line = f.readline()
    [CNN_cluster_list, CNN_cluster_dic] = json.loads(line)
    f.close()

    for j in range(1):
        path_excel = 'E:\\experiment_result\\' + dataset_list[dataset] + '_extension0SF0CNN_' + str(int(time.time())) + str(int(random.uniform(100,900))) + '.xls'
        save_parameter()
        print_parameter()
        train_SPLR(eta)