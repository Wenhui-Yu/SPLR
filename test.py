#-*-coding:utf-8-*-#
#基础模型，SPLR

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

##本部分是参数设置
dataset = 5                         # 选择数据集
eta = 0.1                           # 设置学习率
eta_1 = 0.1
eta_2 = 0.1
I = 200                             # 设置隐特征长度
top_k = [5, 10, 20, 50, 100]        # 设置top k
batch_size_train = 5000             # 设置训练batch size
batch_size_test = 1000              # 设置测试batch size，设置为负值可以测试所有测试样本
lambda_r = 2                        # 设置正则系数
vali_test = 0                       # 0选validate集，1选test集
feat = [0]                          # 选择feature，[x1, x2, ...]的结构，0是CNN，1是AES，2是CH
feature_length = 1000               # 选择不同长度的feature
epoch = 200                         # 设置迭代所有正样本的轮数

def d(x):
    # BPR所用的delta函数
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

def test_SPLR(U, V, M, E, N, F):
    # 测试模型精度
    U = mat(U)
    V = mat(V)
    M = mat(M)
    F = mat(F)
    N = mat(N)
    E = mat(E)
    k_num = len(top_k)
    # k_num长的lists记录F1和NDCG
    F1 = np.zeros(k_num)
    NDCG = np.zeros(k_num)
    num_item = len(Test)
    # 随机选择batch_size_test个测试样本
    for i in range(batch_size_test):
        j = int(math.floor(num_item * random.random()))
        # test的数据格式是[u, [i, i, i, i], [r, r, r]]
        # 为了方便随机选取，将字典改成了列表格式，validate也是
        u = Test[j][0]
        # （测试）正样例是u自带的[i, i, i, i]
        # 为什么叫测试正样例呢？是为了跟训练集里的训练正样例分开
        positive_item = Test[j][1]
        # 通过矩阵相乘为u的每一个商品打分
        score = U[u] * V.T + E[u] * M.T + N[u] * F.T   #get the score of each item
        score = score.tolist()[0]  #mat -> list
        # 排序
        b = zip(score, range(len(score)))
        b.sort(key=lambda x: x[0])
        order = [x[1] for x in b]
        order.reverse()
        # 将训练正样例去除，因为他们已经不（太）可能出现在测试集里了
        train_positive = train_data_aux[u][0]
        for item in train_positive:
            order.remove(item)
        # 为了排除残存的可能性，我们将训练正样本中的物品从测试正样本中也删除
        positive_item = list(set(positive_item) - set(train_positive))
        # 对于每个给定的top_k，都要测试一遍
        for i in range(len(top_k)):
            F1[i] += evaluation_F1(order, top_k[i], positive_item)
            NDCG[i] += evaluation_NDCG(order, top_k[i], positive_item)
    # 累加再除，求平均
    F1 = (F1 / batch_size_test).tolist()
    NDCG = (NDCG / batch_size_test).tolist()
    return F1, NDCG

def train_SPLR(eta):
    # 训练模型
    # 初始化矩阵
    U = np.array([np.array([(random.random() / math.sqrt(I)) for j in range(I)]) for i in range(P)])
    V = np.array([np.array([(random.random() / math.sqrt(I)) for j in range(I)]) for i in range(Q)])
    M = np.array([np.array([(random.random() / math.sqrt(K)) for j in range(K)]) for i in range(Q)])
    N = np.array([np.array([(random.random() / math.sqrt(K)) for j in range(K)]) for i in range(P)])
    e = 100000000000000000000000
    # 输出一个不训练的结果
    print 'iteration ', 0,
    [F1, NDCG] = test_SPLR(U, V, M, E, N, F)
    Fmax = 0
    if F1[0] > Fmax:
        Fmax = F1[0]
    print Fmax, 'F1: ', F1, '  ', 'NDCG1: ', NDCG
    # 写入Excel文件，这些都是为了格式好看
    save_result([' '], [''] * len(top_k), [''] * len(top_k), path_excel)
    save_result('metric', ['F1'] * len(top_k), ['NDCG'] * len(top_k), path_excel)
    save_result('Top_k', top_k, top_k, path_excel)
    save_result([' '], [''] * len(top_k), [''] * len(top_k), path_excel)
    save_result('iteration ' + str(0), F1, NDCG, path_excel)

    # 获得总样本数
    Re = len(train_data)
    # 将样本按batch_size_train为步长划分，划分节点存在bs中
    bs = range(0, Re, batch_size_train)
    bs.append(Re)
    # 开始迭代
    # 迭代所有证样本ep轮
    for ep in range(0, epoch):
        print 'iteration ', ep + 1,
        eta = eta * 0.99
        # 对于每一轮迭代，遍历所有正样本
        for i in range(0, len(bs) - 1):
            if abs(U.sum()) < e:
                # 对于所有证样本，以batch为组输入
                # 初始化dU，dC，用他们记录梯度
                dU = np.zeros((P, I))
                dV = np.zeros((Q, I))
                dM = np.zeros((Q, K))
                dN = np.zeros((P, K))
                for re in range(bs[i], bs[i + 1]):
                    # [u, i, r]
                    p = train_data[re][0]
                    qi = train_data[re][1]
                    xi = np.dot(U[p], V[qi]) + np.dot(E[p], M[qi]) + np.dot(N[p], F[qi])
                    num = 0
                    # 随机生成5个负样本物品，累积梯度
                    # 每个batch有batch_size_train个正样本，每个batch有num个pair，累积batch_size_train*num次
                    while num < 5:
                        qj = int(random.uniform(0, Q))
                        if not (qj in train_data_aux[p][0]):
                            num += 1
                            xj = np.dot(U[p], V[qj]) + np.dot(E[p], M[qj]) + np.dot(N[p], F[qj])
                            xij = xi - xj
                            dU[p] += d(xij) * (V[qi] - V[qj])
                            dV[qi] += d(xij) * U[p]
                            dV[qj] -= d(xij) * U[p]
                            dM[qi] += d(xij) * E[p]
                            dM[qj] -= d(xij) * E[p]
                            dN[p] += d(xij) * (F[qi] - F[qj])

                    neighbor = set()
                    neighbor = neighbor | set(graph_cluster_dic[graph_cluster_list[qi]][0])
                    # neighbor = neighbor | set(CNN_cluster_dic[CNN_cluster_list[qi]][0])
                    # neighbor = neighbor | set(AES_cluster_dic[AES_cluster_list[qi]][0])
                    neighbor = list(neighbor)

                    if len(neighbor) > 5:
                        num = 0
                        while num < 5:
                            qj = neighbor[int(random.uniform(0, len(neighbor) - 0.1))]
                            if not (qj in train_data_aux[p][0]):
                                num += 1
                                xj = np.dot(U[p], V[qj]) + np.dot(E[p], M[qj]) + np.dot(N[p], F[qj])
                                xij = xi - xj
                                dU[p] += eta_1 * d(xij) * (V[qi] - V[qj])
                                dV[qi] += eta_1 * d(xij) * U[p]
                                dV[qj] -= eta_1 * d(xij) * U[p]
                                dM[qi] += eta_1 * d(xij) * E[p]
                                dM[qj] -= eta_1 * d(xij) * E[p]
                                dN[p] += eta_1 * d(xij) * (F[qi] - F[qj])

                        num = 0
                        while num < 5:
                            qj = int(random.uniform(0, Q))
                            if not (qj in train_data_aux[p][0] or qj in neighbor):
                                qi = neighbor[int(random.uniform(0, len(neighbor) - 0.1))]
                                xi = np.dot(U[p], V[qi]) + np.dot(E[p], M[qi]) + np.dot(N[p], F[qi])
                                num += 1
                                xj = np.dot(U[p], V[qj]) + np.dot(E[p], M[qj]) + np.dot(N[p], F[qj])
                                xij = xi - xj
                                dU[p] += eta_2 * d(xij) * (V[qi] - V[qj])
                                dV[qi] += eta_2 * d(xij) * U[p]
                                dV[qj] -= eta_2 * d(xij) * U[p]
                                dM[qi] += eta_2 * d(xij) * E[p]
                                dM[qj] -= eta_2 * d(xij) * E[p]
                                dN[p] += eta_2 * d(xij) * (F[qi] - F[qj])

                # 迭代完一个batch，更新矩阵
                U += eta * (dU - lambda_r * U)
                V += eta * (dV - lambda_r * V)
                M += eta * (dM - lambda_r * M)
                N += eta * (dN - lambda_r * N)
            # 迭代完一轮所有正样本，测试一下精度，输出并写入文件
        if abs(U.sum()) < e:
            [F1, NDCG] = test_SPLR(U, V, M, E, N, F)
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
    # 写入表头，记录实验参数
    dataset_list = ['all', '_Women', '_Men', '_CLothes', '_Shoes', '_Jewelry']
    excel = xlwt.Workbook()
    table = excel.add_sheet('A Test Sheet')
    table.write(0, 0, 'model')
    table.write(0, 2, 'SPLR')
    table.write(1, 0, 'dataset')
    table.write(1, 2, dataset_list[dataset])
    table.write(2, 0, 'eta')
    table.write(2, 2, eta)
    table.write(2, 4, 'eta1')
    table.write(2, 6, eta_1)
    table.write(2, 8, 'eta2')
    table.write(2, 10, eta_2)
    table.write(3, 0, 'I')
    table.write(3, 2, I)
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
    table.write(9, 0, 'feat')
    for i in range(len(feat)):
        table.write(9, 2 + i, feat[i])
    table.write(10, 0, 'fea_len')
    table.write(10, 2, feature_length)
    table.write(16, 0, 'epoch')
    table.write(16, 2, epoch)
    table.write(17, 0, ' ')

    excel.save(path_excel)

def print_parameter():
    # 把所有参数打印出来，使得更加清晰
    print 'model', 'SPLR'
    print 'dataset', dataset
    print 'eta', eta, '    eta1', eta_1, '    eta2', eta_2
    print 'I', I
    print 'top_k', top_k
    print 'batch_size_train', batch_size_train
    print 'batch_size_test', batch_size_test
    print 'lambda_r:', lambda_r
    print 'vali_test', vali_test
    print 'feat', feat
    print 'feature_length', feature_length
    print 'epoch', epoch
    print


'''*************************主函数****************************'''
'''*************************主函数****************************'''
for i in range(1):
    # 数据集列表
    dataset_list = ['', '_Women', '_Men', '_CLothes', '_Shoes', '_Jewelry']
    # 载入数据
    [train_data, train_data_aux, validate_data, test_data, P, Q] = readdata(dataset_list[dataset])
    # 载入feature
    [E, F] = get_feature(dataset)
    '''for i in range(len(E)):
        E[i] = E[i] / math.sqrt(mat(E[i]) * mat(E[i]).T)
    for i in range(len(F)):
        F[i] = F[i] / math.sqrt(mat(F[i]) * mat(F[i]).T)'''

    K = len(E[0])
    # 选择测试集还是验证集
    if vali_test == 0:
        Test = validate_data
    else:
        Test = test_data

    graph_cluster_path = 'E:\dataset\cluster\graph_cluster' + dataset_list[dataset] + '.json'
    CNN_cluster_path = 'E:\dataset\cluster\CNN_cluster' + dataset_list[dataset] + '.json'
    AES_cluster_path = 'E:\dataset\cluster\AES_cluster' + dataset_list[dataset] + '.json'
    f = open(graph_cluster_path, 'r')
    line = f.readline()
    [graph_cluster_list, graph_cluster_dic] = json.loads(line)
    f.close()
    f = open(CNN_cluster_path, 'r')
    line = f.readline()
    [CNN_cluster_list, CNN_cluster_dic] = json.loads(line)
    f.close()
    f = open(AES_cluster_path, 'r')
    line = f.readline()
    [AES_cluster_list, AES_cluster_dic] = json.loads(line)
    f.close()

    for j in range(1):
        path_excel = 'E:\\experiment_result\\' + dataset_list[dataset] + '_SPLR_' + str(int(time.time())) + str(int(random.uniform(100,900))) + '.xls'
        save_parameter()
        print_parameter()
        train_SPLR(eta)