import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

#with open("data/train_set.pickle", "rb") as infile:
#    train_data_set = pickle.load(infile)
with open("data/dev_set.pickle", "rb") as infile:
    dev_data_set = pickle.load(infile)
#with open("data/test_set.pickle", "rb") as infile:
#    test_data_set = pickle.load(infile)

