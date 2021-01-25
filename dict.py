model_dict = {
    'method': ['', 'dag1', 'dag2', 'spbp1', 'spbp2', 'ng', 'hd', 'r', 'pmce', 'pmis', 'bcs', 'cpuh'],
    'r': ['', 'r'],
    'epw': ['', 'epw']
}


def get_model_name(mn_list):
    model_name = 'm' + model_dict['method'][mn_list[0]] + model_dict['r'][mn_list[1]] + model_dict['epw'][mn_list[2]]
    if len(mn_list) > 3:
        model_name += '_' + str(mn_list[-1])

    return model_name


dataset_name_dict = {
    0: 'toy2',
    1: 'email',
    2: 'dnc',
    3: 'Eu',
    4: 'Net',
    5: 'Wiki',
    6: 'Epinions',
    7: 'fb'
}

product_name_dict = {
    1: 'book4',
    2: 'book6'
}

p_dict = {
    1: ([4.5, 7, 15], 0.4),
    2: ([4.5, 7, 15], 0.6)
}

cascade_model_dict = {
    1: 'ic',
    2: 'wc'
}

wallet_distribution_type_dict = {
    0: '',
    1: 'm50e25',
    2: 'm99e96'
}


def generateDistribution(prod_key, wallet_key):
    if prod_key == 1:
        if wallet_key == 1:
            return 7, 11.86
        elif wallet_key == 2:
            return 38, 13.14


epw_dict = {
    0: [1.0, 1.0, 1.0],
    1: [0.5835, 0.5, 0.25],
    2: [0.9946, 0.9908, 0.96]
}