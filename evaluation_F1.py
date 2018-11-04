
def evaluation_F1(order, top_k, positive_item):
    e = 0.00000000000001
    top_k_items = set(order[0: top_k])
    positive_item = set(positive_item)
    precision = len(top_k_items & positive_item) / (len(top_k_items) + e)
    recall = len(top_k_items & positive_item) / (len(positive_item) + e)
    F1 = 2 * precision * recall / (precision + recall + e)
    return F1