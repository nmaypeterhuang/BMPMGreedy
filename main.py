from Model import *

if __name__ == '__main__':
    data_seq = [1, 2, 3, 4, 5]
    cm_seq = [1, 2]
    prod_seq = [1, 2]
    wd_seq = [1, 2]

    for data_key in data_seq:
        for prod_key in prod_seq:
            for cas_key in cm_seq:

                # Model([1, 0, 0], data_key, prod_key, cas_key).model_dag()
                # Model([1, 1, 0], data_key, prod_key, cas_key).model_dag()
                # Model([2, 0, 0], data_key, prod_key, cas_key).model_dag()
                # Model([2, 1, 0], data_key, prod_key, cas_key).model_dag()
                #
                # Model([3, 0, 0], data_key, prod_key, cas_key).model_spbp()
                # Model([3, 1, 0], data_key, prod_key, cas_key).model_spbp()
                # Model([4, 0, 0], data_key, prod_key, cas_key).model_spbp()
                # Model([4, 1, 0], data_key, prod_key, cas_key).model_spbp()
                #
                # for wallet_key in wd_seq:
                #
                #     Model([1, 0, 1], data_key, prod_key, cas_key, wallet_key).model_dag()
                #     Model([1, 1, 1], data_key, prod_key, cas_key, wallet_key).model_dag()
                #     Model([2, 0, 1], data_key, prod_key, cas_key, wallet_key).model_dag()
                #     Model([2, 1, 1], data_key, prod_key, cas_key, wallet_key).model_dag()
                #
                #     Model([3, 0, 1], data_key, prod_key, cas_key, wallet_key).model_spbp()
                #     Model([3, 1, 1], data_key, prod_key, cas_key, wallet_key).model_spbp()
                #     Model([4, 0, 1], data_key, prod_key, cas_key, wallet_key).model_spbp()
                #     Model([4, 1, 1], data_key, prod_key, cas_key, wallet_key).model_spbp()

                for times in range(10):

                    # # Model([5, 0, 0, times], data_key, prod_key, cas_key).model_ng()
                    # Model([5, 1, 0, times], data_key, prod_key, cas_key).model_ng()
                    # Model([6, 0, 0, times], data_key, prod_key, cas_key).model_hd()
                    # Model([7, 0, 0, times], data_key, prod_key, cas_key).model_r()
                    # # Model([8, 0, 0, times], data_key, prod_key, cas_key).model_pmce()
                    # Model([9, 0, 0, times], data_key, prod_key, cas_key).model_pmis()
                    # # Model([10, 0, 0, times], data_key, prod_key, cas_key).model_bcs()
                    Model([11, 0, 0, times], data_key, prod_key, cas_key).model_cpuh()

                    # for wallet_key in wd_seq:
                    #
                    #     Model([5, 0, 1, times], data_key, prod_key, cas_key, wallet_key).model_ng()
                    #     Model([5, 1, 1, times], data_key, prod_key, cas_key, wallet_key).model_ng()
                    #     Model([8, 0, 1, times], data_key, prod_key, cas_key, wallet_key).model_pmce()
                    #     Model([9, 0, 1, times], data_key, prod_key, cas_key, wallet_key).model_pmis()
                    #     Model([10, 1, 1, times], data_key, prod_key, cas_key, wallet_key).model_bcs()