from SeedSelection import *
from Evaluation import *
import time
import copy
import math


class Model:
    def __init__(self, mn_list, data_key, prod_key, cas_key, wallet_key=0):
        self.model_name = get_model_name(mn_list)
        self.r_flag = mn_list[1]
        self.mn_list = mn_list
        self.data_name = dataset_name_dict[data_key]
        self.prod_name = product_name_dict[prod_key]
        self.cas_name = cascade_model_dict[cas_key]
        self.data_key = data_key
        self.prod_key = prod_key
        self.cas_key = cas_key
        self.wallet_type = wallet_distribution_type_dict[wallet_key]
        self.wallet_key = wallet_key
        self.wd_seq = [wd for wd in wallet_distribution_type_dict.keys() if wd != 0]
        self.budget_iteration = [i for i in range(10, 6, -1)]
        self.monte_carlo = 100

    def model_cpuh(self):
        ini = Initialization(self.data_key, self.prod_key, self.cas_key, self.wallet_key)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict()
        product_list, epw_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[i] for i in seed_cost_dict)

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        sscpuh_model = SeedSelectionCPUH(graph_dict, seed_cost_dict, product_list)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget, now_profit = 0.0, 0.0
        discount_rate = 0.5
        seed_set = [set() for _ in range(num_product)]

        wd_seq = [self.wallet_key] if wallet_distribution_type_dict[self.wallet_key] else self.wd_seq
        cp_dict = sscpuh_model.generateCPDict()

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, cp_dict]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, now_profit, seed_set, cp_dict] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ selection\t' + get_model_name(self.mn_list) + ' @ ' + dataset_name_dict[self.data_key] + '_' + cascade_model_dict[self.cas_key] +
                  '\t' + wallet_distribution_type_dict[self.wallet_key] + '_' + product_name_dict[self.prod_key] + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            while now_budget < total_budget and cp_dict:
                mep_k_prod, mep_i_node = max(cp_dict, key=cp_dict.get)
                del cp_dict[(mep_k_prod, mep_i_node)]
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    cp_dict_c = copy.deepcopy(cp_dict)
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), cp_dict_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)
                seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                 str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

                for k_prod in range(num_product):
                    if (k_prod, mep_i_node) in cp_dict:
                        cp_dict[(k_prod, mep_i_node)] *= discount_rate

                for i_node in graph_dict[mep_i_node]:
                    if (mep_k_prod, i_node) in cp_dict:
                        cp_dict[(mep_k_prod, i_node)] *= (1 - graph_dict[mep_i_node][i_node])

                for u_node in graph_dict:
                    if mep_i_node in graph_dict[u_node]:
                        if (mep_k_prod, u_node) in cp_dict:
                            cp_dict[(mep_k_prod, u_node)] -= safe_div(graph_dict[u_node][mep_i_node] * product_list[mep_k_prod][0], sc)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.data_name + '_' + self.cas_name
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_path0 = seed_data_path + '/' + wallet_distribution_type_dict[wd] + '_bi' + str(self.budget_iteration[now_bi_index])
                if not os.path.isdir(seed_data_path0):
                    os.mkdir(seed_data_path0)
                seed_data_file = open(seed_data_path0 + '/' + self.model_name + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_set_sequence[no_data_index] = seed_set_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.mn_list, self.data_key, self.prod_key, self.cas_key)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if wallet_distribution_type_dict[self.wallet_key]:
                eva_model.evaluate(bi, self.wallet_key, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wd in self.wd_seq:
                    eva_model.evaluate(bi, wd, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_dag(self):
        ini = Initialization(self.data_key, self.prod_key, self.cas_key, self.wallet_key)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict()
        product_list, epw_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[i] for i in seed_cost_dict)

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        dag_class = int(list(model_dict['method'][self.mn_list[0]])[-1])
        ssmioa_model = SeedSelectionMIOA(graph_dict, seed_cost_dict, product_list, epw_list, dag_class, self.r_flag)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget, now_profit = 0.0, 0.0
        seed_set = [set() for _ in range(num_product)]

        wd_seq = [self.wallet_key] if wallet_distribution_type_dict[self.wallet_key] else self.wd_seq
        mioa_dict = ssmioa_model.generateMIOA()
        celf_heap = ssmioa_model.generateCelfHeap(mioa_dict)

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, celf_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ selection\t' + self.model_name + ' @ ' + self.data_name + '_' + self.cas_name +
                  '\t' + self.wallet_type + '_' + self.prod_name + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            celf_heap_c = []
            while now_budget < total_budget and celf_heap:
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    celf_heap_c = copy.deepcopy(celf_heap)
                seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), celf_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    now_profit = round(now_profit + (mep_mg * (sc if self.r_flag else 1.0)), 4)
                    seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                     str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    dag_dict = [{} for _ in range(num_product)]
                    if dag_class == 1:
                        dag_dict = ssmioa_model.generateDAG1(mioa_dict, seed_set_t)
                    elif dag_class == 2:
                        dag_dict = ssmioa_model.generateDAG2(mioa_dict, seed_set_t)
                    ep_t = ssmioa_model.calculateExpectedProfit(dag_dict, seed_set_t)
                    mg_t = safe_div(round(ep_t - now_profit, 4), sc if self.r_flag else 1.0)
                    flag_t = seed_set_length

                    if mg_t > 0:
                        celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                        heap.heappush_max(celf_heap, celf_item_t)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.data_name + '_' + self.cas_name
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_path0 = seed_data_path + '/' + wallet_distribution_type_dict[wd] + '_bi' + str(self.budget_iteration[now_bi_index])
                if not os.path.isdir(seed_data_path0):
                    os.mkdir(seed_data_path0)
                seed_data_file = open(seed_data_path0 + '/' + self.model_name + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_set_sequence[no_data_index] = seed_set_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.mn_list, self.data_key, self.prod_key, self.cas_key)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if wallet_distribution_type_dict[self.wallet_key]:
                eva_model.evaluate(bi, self.wallet_key, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wd in self.wd_seq:
                    eva_model.evaluate(bi, wd, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_spbp(self):
        ini = Initialization(self.data_key, self.prod_key, self.cas_key, self.wallet_key)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict()
        product_list, epw_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[i] for i in seed_cost_dict)

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        dag_class = int(list(model_dict['method'][self.mn_list[0]])[-1])
        ssmioa_model = SeedSelectionMIOA(graph_dict, seed_cost_dict, product_list, epw_list, dag_class, self.r_flag)
        ssspbp_model = SeedSelectionSPBP(graph_dict, seed_cost_dict, product_list, epw_list, dag_class, self.r_flag)

        for now_b_iter in self.budget_iteration:
            ss_start_time = time.time()
            now_budget, now_profit = 0.0, 0.0
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            seed_set = [set() for _ in range(num_product)]

            wd_seq = [self.wallet_key] if wallet_distribution_type_dict[self.wallet_key] else self.wd_seq
            mioa_dict, ps_dict = ssspbp_model.generateMIOA()
            celf_dict, max_s = ssspbp_model.generateCelfDict(mioa_dict, total_budget)

            seed_data = ['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']
            print('@ selection\t' + get_model_name(self.mn_list) + ' @ ' + dataset_name_dict[self.data_key] + '_' + cascade_model_dict[self.cas_key] +
                  '\t' + wallet_distribution_type_dict[self.wallet_key] + '_' + product_name_dict[self.prod_key] + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            while now_budget < total_budget and celf_dict:
                mep_k_prod, mep_i_node = max(celf_dict, key=celf_dict.get)
                mep_mg = max(celf_dict.values())
                del celf_dict[(mep_k_prod, mep_i_node)]
                sc = seed_cost_dict[mep_i_node]

                if round(now_budget + sc, 4) > total_budget:
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)
                now_profit = round(now_profit + (mep_mg * (sc if self.r_flag else 1.0)), 4)
                seed_data.append(str(round(time.time() - ss_start_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                 str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

                delta_max = 0.0
                if mep_i_node in ps_dict[mep_k_prod]:
                    for (k, i) in ps_dict[mep_k_prod][mep_i_node]:
                        if i in seed_set[k]:
                            continue

                        if (k, i) not in celf_dict:
                            continue

                        if celf_dict[(k, i)] > delta_max:
                            seed_set_t = copy.deepcopy(seed_set)
                            seed_set_t[mep_k_prod].add(mep_i_node)
                            dag_dict = [{} for _ in range(num_product)]
                            if dag_class == 1:
                                dag_dict = ssmioa_model.generateDAG1(mioa_dict, seed_set_t)
                            elif dag_class == 2:
                                dag_dict = ssmioa_model.generateDAG2(mioa_dict, seed_set_t)
                            ep_t = ssmioa_model.calculateExpectedProfit(dag_dict, seed_set_t)
                            mg_t = round(ep_t - now_profit, 4)
                            mg_t = safe_div(mg_t, sc) if self.r_flag else mg_t
                            celf_dict[(k, i)] = mg_t
                            delta_max = mg_t if mg_t > delta_max else delta_max

            if max_s[0] > now_profit and max_s[-1] != '-1':
                seed_set = [set() for _ in range(num_product)]
                seed_set[max_s[1]].add(max_s[2])

            ss_time = round(time.time() - ss_start_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.data_name + '_' + self.cas_name
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_path0 = seed_data_path + '/' + wallet_distribution_type_dict[wd] + '_bi' + str(self.budget_iteration[now_bi_index])
                if not os.path.isdir(seed_data_path0):
                    os.mkdir(seed_data_path0)
                seed_data_file = open(seed_data_path0 + '/' + self.model_name + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_set_sequence[no_data_index] = seed_set_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.mn_list, self.data_key, self.prod_key, self.cas_key)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if wallet_distribution_type_dict[self.wallet_key]:
                eva_model.evaluate(bi, self.wallet_key, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wd in self.wd_seq:
                    eva_model.evaluate(bi, wd, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_ng(self):
        ini = Initialization(self.data_key, self.prod_key, self.cas_key, self.wallet_key)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict()
        product_list, epw_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[i] for i in seed_cost_dict)

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, epw_list, self.r_flag)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget, now_profit = 0.0, 0.0
        seed_set = [set() for _ in range(num_product)]

        wd_seq = [self.wallet_key] if wallet_distribution_type_dict[self.wallet_key] else self.wd_seq
        celf_heap = ssng_model.generateCelfHeap()

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, celf_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ selection\t' + get_model_name(self.mn_list) + ' @ ' + dataset_name_dict[self.data_key] + '_' + cascade_model_dict[self.cas_key] +
                  '\t' + wallet_distribution_type_dict[self.wallet_key] + '_' + product_name_dict[self.prod_key] + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            celf_heap_c = []
            while now_budget < total_budget and celf_heap:
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    celf_heap_c = copy.deepcopy(celf_heap)
                seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), celf_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    now_profit = ssng_model.getSeedSetProfit(seed_set)
                    seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                     str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_t = ssng_model.getSeedSetProfit(seed_set_t)
                    mg_t = round(ep_t - now_profit, 4)
                    if self.r_flag:
                        mg_t = safe_div(mg_t, sc)
                    flag_t = seed_set_length

                    if mg_t > 0:
                        celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                        heap.heappush_max(celf_heap, celf_item_t)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.data_name + '_' + self.cas_name
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_path0 = seed_data_path + '/' + wallet_distribution_type_dict[wd] + '_bi' + str(self.budget_iteration[now_bi_index])
                if not os.path.isdir(seed_data_path0):
                    os.mkdir(seed_data_path0)
                seed_data_file = open(seed_data_path0 + '/' + self.model_name + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_set_sequence[no_data_index] = seed_set_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.mn_list, self.data_key, self.prod_key, self.cas_key)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if wallet_distribution_type_dict[self.wallet_key]:
                eva_model.evaluate(bi, self.wallet_key, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wd in self.wd_seq:
                    eva_model.evaluate(bi, wd, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_hd(self):
        ini = Initialization(self.data_key, self.prod_key, self.cas_key, self.wallet_key)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict()
        product_list, epw_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[i] for i in seed_cost_dict)

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        sshd_model = SeedSelectionHD(graph_dict, product_list)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget = 0.0
        seed_set = [set() for _ in range(num_product)]

        wd_seq = [self.wallet_key] if wallet_distribution_type_dict[self.wallet_key] else self.wd_seq
        degree_heap = sshd_model.generateDegreeHeap()

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, seed_set, degree_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, seed_set, degree_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ selection\t' + get_model_name(self.mn_list) + ' @ ' + dataset_name_dict[self.data_key] + '_' + cascade_model_dict[self.cas_key] +
                  '\t' + wallet_distribution_type_dict[self.wallet_key] + '_' + product_name_dict[self.prod_key] + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            degree_heap_c = []
            while now_budget < total_budget and degree_heap:
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    degree_heap_c = copy.deepcopy(degree_heap)

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, copy.deepcopy(seed_set), degree_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)
                seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                 str(now_budget) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.data_name + '_' + self.cas_name
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_path0 = seed_data_path + '/' + wallet_distribution_type_dict[wd] + '_bi' + str(self.budget_iteration[now_bi_index])
                if not os.path.isdir(seed_data_path0):
                    os.mkdir(seed_data_path0)
                seed_data_file = open(seed_data_path0 + '/' + self.model_name + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_set_sequence[no_data_index] = seed_set_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.mn_list, self.data_key, self.prod_key, self.cas_key)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if wallet_distribution_type_dict[self.wallet_key]:
                eva_model.evaluate(bi, self.wallet_key, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wd in self.wd_seq:
                    eva_model.evaluate(bi, wd, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_r(self):
        ini = Initialization(self.data_key, self.prod_key, self.cas_key, self.wallet_key)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict()
        product_list, epw_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[i] for i in seed_cost_dict)

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget = 0.0
        seed_set = [set() for _ in range(num_product)]

        wd_seq = [self.wallet_key] if wallet_distribution_type_dict[self.wallet_key] else self.wd_seq
        random_node_list = [(k, i) for i in graph_dict for k in range(num_product)]
        random.shuffle(random_node_list)

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, seed_set, random_node_list]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, seed_set, random_node_list] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ selection\t' + get_model_name(self.mn_list) + ' @ ' + dataset_name_dict[self.data_key] + '_' + cascade_model_dict[self.cas_key] +
                  '\t' + wallet_distribution_type_dict[self.wallet_key] + '_' + product_name_dict[self.prod_key] + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            random_node_list_c = []
            while now_budget < total_budget and random_node_list:
                mep_item = random_node_list.pop(0)
                mep_k_prod, mep_i_node = mep_item
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    random_node_list_c = copy.deepcopy(random_node_list)

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, copy.deepcopy(seed_set), random_node_list_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)
                seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                 str(now_budget) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.data_name + '_' + self.cas_name
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_path0 = seed_data_path + '/' + wallet_distribution_type_dict[wd] + '_bi' + str(self.budget_iteration[now_bi_index])
                if not os.path.isdir(seed_data_path0):
                    os.mkdir(seed_data_path0)
                seed_data_file = open(seed_data_path0 + '/' + self.model_name + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_set_sequence[no_data_index] = seed_set_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.mn_list, self.data_key, self.prod_key, self.cas_key)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if wallet_distribution_type_dict[self.wallet_key]:
                eva_model.evaluate(bi, self.wallet_key, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wd in self.wd_seq:
                    eva_model.evaluate(bi, wd, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_pmce(self):
        ini = Initialization(self.data_key, self.prod_key, self.cas_key, self.wallet_key)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict()
        product_list, epw_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[i] for i in seed_cost_dict)

        profit_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        sspmce_model = SeedSelectionPMCE(graph_dict, seed_cost_dict, product_list, epw_list)
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, epw_list, None)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget, now_profit = 0.0, 0.0
        seed_set = [set() for _ in range(num_product)]

        wd_seq = [self.wallet_key] if wallet_distribution_type_dict[self.wallet_key] else self.wd_seq
        celf_heap, celf_heap2 = sspmce_model.generateCelfHeap()

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, celf_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ selection\t' + get_model_name(self.mn_list) + ' @ ' + dataset_name_dict[self.data_key] + '_' + cascade_model_dict[self.cas_key] +
                  '\t' + wallet_distribution_type_dict[self.wallet_key] + '_' + product_name_dict[self.prod_key] + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            celf_heap_c = []
            while now_budget < total_budget and celf_heap:
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    celf_heap_c = copy.deepcopy(celf_heap)
                seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), celf_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    now_profit = ssng_model.getSeedSetProfit(seed_set)
                    seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                     str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_t = ssng_model.getSeedSetProfit(seed_set_t)
                    mg_t = safe_div(round(ep_t - now_profit, 4), pow(sc, 2))
                    flag_t = seed_set_length

                    if mg_t > 0:
                        celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                        heap.heappush_max(celf_heap, celf_item_t)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            profit_sequence[now_bi_index] = now_profit
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget, now_profit = 0.0, 0.0
        seed_set = [set() for _ in range(num_product)]

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, celf_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ selection\t' + get_model_name(self.mn_list) + ' @ ' + dataset_name_dict[self.data_key] + '_' + cascade_model_dict[self.cas_key] +
                  '\t' + wallet_distribution_type_dict[self.wallet_key] + '_' + product_name_dict[self.prod_key] + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            celf_heap_c = []
            while now_budget < total_budget and celf_heap:
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    celf_heap_c = copy.deepcopy(celf_heap)
                seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), celf_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    now_profit = sspmce_model.getSeedSetProfit(seed_set)
                    seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                     str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_t = ssng_model.getSeedSetProfit(seed_set_t)
                    mg_t = round(ep_t - now_profit, 4)
                    flag_t = seed_set_length

                    if mg_t > 0:
                        celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                        heap.heappush_max(celf_heap, celf_item_t)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            if now_profit > profit_sequence[now_bi_index]:
                profit_sequence[now_bi_index] = now_profit
                seed_set_sequence[now_bi_index] = seed_set
                seed_data_sequence[now_bi_index] = seed_data
            ss_time_sequence[now_bi_index] += ss_time

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.data_name + '_' + self.cas_name
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_path0 = seed_data_path + '/' + wallet_distribution_type_dict[wd] + '_bi' + str(self.budget_iteration[now_bi_index])
                if not os.path.isdir(seed_data_path0):
                    os.mkdir(seed_data_path0)
                seed_data_file = open(seed_data_path0 + '/' + self.model_name + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_set_sequence[no_data_index] = seed_set_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.mn_list, self.data_key, self.prod_key, self.cas_key)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if wallet_distribution_type_dict[self.wallet_key]:
                eva_model.evaluate(bi, self.wallet_key, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wd in self.wd_seq:
                    eva_model.evaluate(bi, wd, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_pmis(self):
        ini = Initialization(self.data_key, self.prod_key, self.cas_key, self.wallet_key)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict()
        product_list, epw_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[i] for i in seed_cost_dict)

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]

        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, epw_list, True)
        sspmis_model = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, epw_list)

        ss_start_time = time.time()
        celf_heap_o = sspmis_model.generateCelfHeap()

        ss_acc_time = round(time.time() - ss_start_time, 4)
        for now_b_iter in self.budget_iteration:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            celf_heap = copy.deepcopy(celf_heap_o)
            print('@ selection\t' + get_model_name(self.mn_list) + ' @ ' + dataset_name_dict[self.data_key] + '_' + cascade_model_dict[self.cas_key] +
                  '\t' + wallet_distribution_type_dict[self.wallet_key] + '_' + product_name_dict[self.prod_key] + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            # -- initialization for each sample --
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            s_matrix, c_matrix = [[set() for _ in range(num_product)]], [0.0]

            while now_budget < total_budget and celf_heap:
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                if round(now_budget + sc, 4) > total_budget:
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    now_profit = ssng_model.getSeedSetProfit(seed_set)
                    s_matrix.append(copy.deepcopy(seed_set))
                    c_matrix.append(now_budget)
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_t = ssng_model.getSeedSetProfit(seed_set_t)
                    mg_t = round(ep_t - now_profit, 4)
                    flag_t = seed_set_length

                    if mg_t > 0:
                        celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                        heap.heappush_max(celf_heap, celf_item_t)

            seed_set = sspmis_model.solveMCPK(total_budget, [s_matrix] * num_product, [c_matrix] * num_product)
            now_budget = sum(seed_cost_dict[i] for k in range(num_product) for i in seed_set[k])

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_set

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_set_sequence[no_data_index] = seed_set_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.mn_list, self.data_key, self.prod_key, self.cas_key)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if wallet_distribution_type_dict[self.wallet_key]:
                eva_model.evaluate(bi, self.wallet_key, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wd in self.wd_seq:
                    eva_model.evaluate(bi, wd, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_bcs(self):
        ini = Initialization(self.data_key, self.prod_key, self.cas_key, self.wallet_key)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict()
        product_list, epw_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[i] for i in seed_cost_dict)

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]

        ssbcs_model = SeedSelectionBCS(graph_dict, seed_cost_dict, product_list, epw_list)

        ss_start_time = time.time()
        celf_heap_list_o = ssbcs_model.generateCelfHeap()

        ss_acc_time = round(time.time() - ss_start_time, 4)
        for now_b_iter in self.budget_iteration:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            celf_heap_list = copy.deepcopy(celf_heap_list_o)
            print('@ selection\t' + get_model_name(self.mn_list) + ' @ ' + dataset_name_dict[self.data_key] + '_' + cascade_model_dict[self.cas_key] +
                  '\t' + wallet_distribution_type_dict[self.wallet_key] + '_' + product_name_dict[self.prod_key] + '_bi' + str(now_b_iter) + ', budget = ' + str(total_budget))

            seed_set_list = []
            while celf_heap_list:
                celf_heap = celf_heap_list.pop()
                now_budget, now_profit = 0.0, 0.0
                seed_set = [set() for _ in range(num_product)]

                while now_budget < total_budget and celf_heap:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                    sc = seed_cost_dict[mep_i_node]
                    seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                    if round(now_budget + sc, 4) > total_budget:
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(now_profit + mep_mg * (sc if len(celf_heap_list) else 1.0), 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_t = ssbcs_model.getSeedSetProfit(seed_set_t)
                        mg_t = round(ep_t - now_profit, 4)
                        if len(celf_heap_list):
                            mg_t = safe_div(mg_t, sc)
                        flag_t = seed_set_length

                        if mg_t > 0:
                            celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                            heap.heappush_max(celf_heap, celf_item_t)

                seed_set_list.insert(0, seed_set)

            final_seed_set = copy.deepcopy(seed_set_list[0])
            final_bud = sum(seed_cost_dict[i] for k in range(num_product) for i in final_seed_set[k])
            final_ep = ssbcs_model.getSeedSetProfit(seed_set_list[0])
            for k in range(num_product):
                Handbill_counter = 0
                AnnealingScheduleT, detT = 1000000, 1000
                for s in seed_set_list[0][k]:
                    # -- first level: replace billboard seed by handbill seed --
                    final_seed_set_t = copy.deepcopy(final_seed_set)
                    final_seed_set_t[k].remove(s)
                    final_bud_t = final_bud - seed_cost_dict[s]
                    Handbill_seed_set = set((k, i) for k in range(num_product) for i in seed_set_list[1][k] if i not in final_seed_set_t[k])
                    if Handbill_seed_set:
                        min_Handbill_cost = min(seed_cost_dict[Handbill_item[1]] for Handbill_item in Handbill_seed_set)
                        while total_budget - final_bud_t >= min_Handbill_cost and Handbill_seed_set:
                            k_prod, i_node = Handbill_seed_set.pop()
                            if seed_cost_dict[i_node] <= total_budget - final_bud_t:
                                final_seed_set_t[k_prod].add(i_node)
                                final_bud_t += seed_cost_dict[i_node]
                                Handbill_counter += 1
                        final_ep_t = ssbcs_model.getSeedSetProfit(final_seed_set_t)
                        final_mg_t = final_ep_t - final_ep
                        # -- second level: replace handbill seed by handbill seed --
                        if final_mg_t >= 0 or math.exp(safe_div(final_mg_t, AnnealingScheduleT)) > random.random():
                            final_seed_set = final_seed_set_t
                            final_bud = final_bud_t
                            final_ep = final_ep_t
                            for q in range(min(Handbill_counter, 10)):
                                final_seed_set_t = copy.deepcopy(final_seed_set)
                                final_Handbill_seed_set = set((k, i) for k in range(num_product) for i in final_seed_set_t[k] if i in seed_set_list[1][k])
                                if final_Handbill_seed_set:
                                    k_prod, i_node = final_Handbill_seed_set.pop()
                                    final_seed_set_t[k_prod].remove(i_node)
                                    final_bud_t = final_bud - seed_cost_dict[i_node]
                                    Handbill_seed_set = set((k, i) for k in range(num_product) for i in seed_set_list[1][k] if i not in final_seed_set_t[k])
                                    min_Handbill_cost = min(seed_cost_dict[Handbill_item[1]] for Handbill_item in Handbill_seed_set)
                                    while total_budget - final_bud_t >= min_Handbill_cost and Handbill_seed_set:
                                        k_prod, i_node = Handbill_seed_set.pop()
                                        if seed_cost_dict[i_node] <= total_budget - final_bud_t:
                                            final_seed_set_t[k_prod].add(i_node)
                                            final_bud_t += seed_cost_dict[i_node]
                                    final_ep_t = ssbcs_model.getSeedSetProfit(final_seed_set_t)
                                    final_mg_t = final_ep_t - final_ep
                                    if final_mg_t >= 0 or math.exp(safe_div(final_mg_t, AnnealingScheduleT)) > random.random():
                                        final_seed_set = final_seed_set_t
                                        final_bud = final_bud_t
                                        final_ep = final_ep_t

                    AnnealingScheduleT -= detT
            seed_set = copy.deepcopy(final_seed_set)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(final_bud) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = final_seed_set

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_set_sequence[no_data_index] = seed_set_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.mn_list, self.data_key, self.prod_key, self.cas_key)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if wallet_distribution_type_dict[self.wallet_key]:
                eva_model.evaluate(bi, self.wallet_key, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wd in self.wd_seq:
                    eva_model.evaluate(bi, wd, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])