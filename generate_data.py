import csv
from subset import *

data_header = ['n', 'k', 'total']

def generate_dict_subsets():
    for i in range(2, 12):
        generateSubsets(i)
    data = dict_subsets
    with open('data/dict_subsets1.csv', 'w') as file:
        writer = csv.writer(file)
        for k, v in data.items():
            writer.writerow([k, v])

    for i in range(12, 13):
        generateSubsets(i)
    data = dict_subsets
    with open('data/dict_subsets2.csv', 'w') as file:
        writer = csv.writer(file)
        for k, v in data.items():
            writer.writerow([k, v])

        

def generate_data(nlist, klist):
    data = []
    for n in nlist:
        for k in klist:
            data.append([n, k, subsetsDivisible(n, k)])
    return data

def generate_train_dev_test():
    generate_train()
    generate_dev()
    generate_test()

def generate_train():
    data = 1 # FIXME
    with open('data/train_set.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(data_header)
        writer.writerows(data)

def generate_dev():
    data = 1 # FIXME
    with open('data/dev_set.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(data_header)
        writer.writerows(data)

def generate_test():
    data = 1 # FIXME
    with open('data/test_set.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(data_header)
        writer.writerows(data)

generate_dict_subsets()