from numpy import *

def evaluation_NDCG(order, top_k, positive_item):
    top_k_item = order[0: top_k]
    e = 0.0000000001
    Z_u = 0
    temp = 0
    for i in range(0, top_k):
        Z_u += 1 / log2(i + 2)
        if top_k_item[i] in positive_item:
            temp += 1 / log2(i + 2)
    NDCG = temp / (Z_u + e)
    return NDCG