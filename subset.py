from xml.etree.ElementTree import tostring
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
from itertools import combinations

dict_subsets = {}

reader = csv.reader(open('data/dict_subsets1.csv', 'r'))
for row in reader:
   k, v = row
   dict_subsets[k] = v
reader = csv.reader(open('data/dict_subsets2.csv', 'r'))
for row in reader:
   k, v = row
   dict_subsets[k] = v

print(dict_subsets)

def generateSubsets(n):
    """Returns a list of subsets of the set {1, 2, 3, ..., n}.

    >>> generateSubsets(3)
    [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]    
    >>> generateSubsets(1)
    [[], [1]]
    >>> generateSubsets(0)
    [[]]
    >>> len(generateSubsets(4)) # 2^4 = 16
    16 
    """

    set = [x for x in range(n + 1)][1:]
    subsets = []
	
    for i in range(0, len(set) + 1):
        temp = [list(x) for x in combinations(set, i)]
        if len(temp) > 0:
            subsets.extend(temp)

    # Save this subset so it doesn't need to be generated again
    global dict_subsets
    dict_subsets[n] = subsets

    return subsets



def subsetsDivisible(n, k):
    """Returns the number of subsets of the set {1, 2, 3, ..., n} that
    have a sum that is divisible by k.
    
    >>> subsetsDivisible(3, 2)
    4
    >>> subsetsDivisible(7, 3)
    44
    >>> subsetsDivisible(6, 5)
    14
    """
    if n in dict_subsets:
        subsets = dict_subsets[n]
    else:
        subsets = generateSubsets(n)

    counter = 0

    for s in subsets:
        if sum(s) % k == 0:
            counter += 1
    
    return counter