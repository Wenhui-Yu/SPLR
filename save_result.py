from xlrd import open_workbook
from xlutils.copy import copy
def save_result(intro, F1, NDCG, path):
    rexcel = open_workbook(path)
    rows = rexcel.sheets()[0].nrows
    excel = copy(rexcel)
    table = excel.get_sheet(0)
    row = rows
    table.write(row, 0, intro)
    #table.write(row, 2, 'F1')
    for i in range(len(F1)):
        table.write(row, i + 3, F1[i])
    #table.write(row, len(F1) + 4, 'NDCG')
    for i in range(len(NDCG)):
        table.write(row, i + len(F1) + 5, NDCG[i])
    excel.save(path)
