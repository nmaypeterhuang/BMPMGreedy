from collectFile import *
import xlwings as xw

data_seq = [1, 2, 3, 4, 5]
cm_seq = [1, 2]
prod_seq = [1, 2]
wd_seq = [1, 2]
model_seq2 = [
    [1, 0, 1], [1, 1, 1], [2, 0, 1], [2, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [1, 1, 0], [1, 1, 1], [2, 1, 0], [2, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [1, 1, 1], [2, 1, 1], [3, 1, 0], [4, 1, 0], [5, 1, 0], [11, 0, 0], [6, 0, 0], [7, 0, 0], [0, 0, 0]
]

for data_key in data_seq:
    data_name = dataset_name_dict[data_key]
    for cm_key in cm_seq:
        cm_name = cascade_model_dict[cm_key]
        profit2_list, time2_list = [], []
        for wallet_key in wd_seq:
            wallet_type = wallet_distribution_type_dict[wallet_key]
            for prod_key in prod_seq:
                prod_name = product_name_dict[prod_key]
                for bi in range(10, 6, -1):
                    profit2, time2 = [], []
                    for mn_list in model_seq2:
                        model_name = get_model_name(mn_list)
                        try:
                            result_name = 'resultT/' + \
                                          data_name + '_' + cm_name + '/' + \
                                          wallet_type + '_' + prod_name + '_bi' + str(bi) + '/' + \
                                          model_name + '.txt'

                            with open(result_name) as f:
                                p = 0.0
                                for lnum, line in enumerate(f):
                                    if lnum < 2 or lnum == 3:
                                        continue
                                    elif lnum == 2:
                                        (l) = line.split()
                                        t = l[-1]
                                        time2.append(t)
                                    elif lnum == 4:
                                        (l) = line.split()
                                        p = float(l[-1])
                                    elif lnum == 5:
                                        (l) = line.split()
                                        c = float(l[-1])
                                        profit2.append(str(round(p - c, 4)))
                                    else:
                                        break
                        except FileNotFoundError:
                            profit2.append('')
                            time2.append('')
                    profit2_list.append(profit2)
                    time2_list.append(time2)

        result_path = 'resultT/comparison_v2.xlsx'
        wb = xw.Book(result_path)
        sheet_name = data_name + '_' + cm_name
        sheet = wb.sheets[sheet_name]
        sheet.cells(9, "D").value = profit2_list
        sheet.cells(26, "D").value = time2_list