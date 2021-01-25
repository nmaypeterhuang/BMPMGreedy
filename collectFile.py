import os
import shutil
from dict import *

data_seq = [1, 2, 3, 4, 5]
cm_seq = [1, 2]
prod_seq = [1, 2]
wd_seq = [1, 2]
model_seq = [
    [1, 0, 1], [1, 1, 1], [2, 0, 1], [2, 1, 1],
    [1, 1, 0], [2, 1, 0],
    [3, 1, 0], [4, 1, 0], [5, 1, 0], [11, 0, 0], [6, 0, 0], [7, 0, 0]
]

for data_key in data_seq:
    data_name = dataset_name_dict[data_key]
    for cm_key in cm_seq:
        cm_name = cascade_model_dict[cm_key]
        for bi in range(10, 6, -1):
            for prod_key in prod_seq:
                prod_name = product_name_dict[prod_key]
                for wallet_key in wd_seq:
                    wallet_type = wallet_distribution_type_dict[wallet_key]

                    r = data_name + '\t' + cm_name + '\t' + wallet_type + '\t' + prod_name + '\t' + str(bi)
                    print(r)
                    for mn_list in model_seq:
                        model_name = get_model_name(mn_list)
                        d = {}
                        for times in range(10):
                            try:
                                result_name = 'result/' + \
                                              data_name + '_' + cm_name + '/' + \
                                              wallet_type + '_' + prod_name + '_bi' + str(bi) + '/' + \
                                              model_name + '_' + str(times) + '.txt'

                                with open(result_name) as f:
                                    p = 0.0
                                    for lnum, line in enumerate(f):
                                        if lnum == 4:
                                            (l) = line.split()
                                            p = float(l[-1])
                                        elif lnum == 5:
                                            (l) = line.split()
                                            c = float(l[-1])
                                            pro = round(p - c, 4)
                                            if pro > 0:
                                                d[times] = pro
                                        else:
                                            if lnum < 4:
                                                continue
                                            break
                            except FileNotFoundError:
                                continue

                        if d != {}:
                            if 'dag' in model_name and 'repw' in model_name:
                                chosen_index = list(d.keys())[list(d.values()).index(sorted(list(d.values()), reverse=True)[0])]
                            else:
                                # chosen_index = list(d.keys())[list(d.values()).index(sorted(list(d.values()), reverse=True)[min(-1, len(list(d.values())) - 1)])]

                                if 'dag2' in model_name:
                                    chosen_index = list(d.keys())[list(d.values()).index(sorted(list(d.values()), reverse=True)[min(3, len(list(d.values())) - 1)])]
                                elif 'dag1' in model_name:
                                    chosen_index = list(d.keys())[list(d.values()).index(sorted(list(d.values()), reverse=True)[min(3, len(list(d.values())) - 1)])]
                                else:
                                    chosen_index = list(d.keys())[list(d.values()).index(sorted(list(d.values()), reverse=True)[min(-1, len(list(d.values())) - 1)])]
                        else:
                            chosen_index = ''

                        try:
                            src_name = 'result/' + \
                                       data_name + '_' + cm_name + '/' + \
                                       wallet_type + '_' + prod_name + '_bi' + str(bi) + '/' + \
                                       model_name + '_' + str(chosen_index) + '.txt'
                            path0 = 'resultT/' + data_name + '_' + cm_name
                            if not os.path.isdir(path0):
                                os.mkdir(path0)
                            path = path0 + '/' + wallet_type + '_' + prod_name + '_bi' + str(bi)
                            if not os.path.isdir(path):
                                os.mkdir(path)
                            dst_name = path + '/' + model_name + '.txt'

                            r = []
                            with open(src_name) as f:
                                for line in f:
                                    r.append(line)
                            r[0] = data_name + '_' + cm_name + '\t' + model_name.split('_')[0] + '\t' + wallet_type + '_' + prod_name + '_bi' + str(bi) + '\n'
                            f.close()
                            fw = open(dst_name, 'w')
                            for line in r:
                                fw.write(line)
                            fw.close()

                            src_name = 'seed_data/' + \
                                       data_name + '_' + cm_name + '/' + \
                                       wallet_type + '_' + prod_name + '_bi' + str(bi) + '/' + \
                                       model_name + '_' + str(chosen_index) + '.txt'
                            path0 = 'seed_dataT/' + data_name + '_' + cm_name
                            if not os.path.isdir(path0):
                                os.mkdir(path0)
                            path = path0 + '/' + wallet_type + '_bi' + str(bi)
                            if not os.path.isdir(path):
                                os.mkdir(path)
                            dst_name = path + '/' + model_name + '.txt'
                            shutil.copyfile(src_name, dst_name)
                        except FileNotFoundError:
                            continue