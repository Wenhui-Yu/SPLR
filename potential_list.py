def potential_list(cluster_list, train_data_aux, N, M):
    item_index_list = []
    for i in range(M):
        item_index_list.append(0)
    for cluster in range(len(cluster_list)):
        for items in cluster_list[cluster]:
            item_index_list[items] = cluster
    pot_list = []
    for i in range(N):
        pot_list.append([])
    pot_index_list = list(pot_list)
    for u in range(N):
        for i in train_data_aux[u][0]:
            #print N,M,u,i
            pot_index_list[u] += [item_index_list[i]]
    for u in range(N):
        pot_index_list[u] = list(set(pot_index_list[u]))
    for u in range(N):
        for index in pot_index_list[u]:
            pot_list[u] += cluster_list[index]
    for u in range(N):
        for i in train_data_aux[u][0]:
            pot_list[u].remove(i)
    return pot_list

