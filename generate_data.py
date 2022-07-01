import random
import pickle
from subset import *

data_header = ['n', 'k', 'total']

def generate_dict_subsets():
    for i in range(2, 26):
        generateSubsets(i)
    data = dict_subsets

    with open("data/dict_subsets.pickle", "wb") as outfile:
        pickle.dump(data, outfile)

        

def generate_data(nlist, klist, set):
    data = []
    total_computations = len(nlist) * len(nlist)
    computations_completed = 0
    increment = 0.1
    for n in nlist:
        for k in klist:
            data.append([n, k, subsetsDivisible(n, k)])
            computations_completed += 1
            if computations_completed / total_computations >= increment:
                print(set + ": " + str(computations_completed / total_computations))
                increment += 0.1
    return data

def generate_train_dev_test():
    generate_train()
    generate_dev()
    generate_test()

def generate_train():
    n_values = random.sample(range(2, 26), 20)
    k_values = random.sample(range(2,250), 200)
    data = generate_data(n_values, k_values, "train")
    with open("data/train_set.pickle", "wb") as outfile:
        pickle.dump(data, outfile)

def generate_dev():
    n_values = random.sample(range(2, 26), 5)
    k_values = random.sample(range(2,250), 50)
    data = generate_data(n_values, k_values, "dev")
    with open("data/dev_set.pickle", "wb") as outfile:
        pickle.dump(data, outfile)

def generate_test():
    n_values = random.sample(range(2, 26), 5)
    k_values = random.sample(range(2,250), 50)
    data = generate_data(n_values, k_values, "test")
    with open("data/test_set.pickle", "wb") as outfile:
        pickle.dump(data, outfile)

print("Begin generating dict_subsets")
generate_dict_subsets()

print("Begin generating data")
generate_train_dev_test()