from xml.etree.ElementTree import tostring
import matplotlib.pyplot as plt
import numpy as np
import math

from itertools import combinations


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
    counter = 0

    for s in generateSubsets(n):
        if sum(s) % k == 0:
            counter += 1
    
    return counter

def plotBoth(n, kMax):

    plotSample(n, kMax)
    plotConjecture(n, kMax)

    plt.xlabel('k')
    plt.ylabel('number of subsets divisible by k')
    plt.title("n = " + str(n))

    plt.show()

    plt.style.use('_mpl-gallery')

def plotSample(n, kMax):
    plt.plot([x for x in range(kMax + 1)][1:], [subsetsDivisible(n, x) for x in range(kMax + 1)[1:]])

def plotConjecture(n, kMax):
    plt.plot([x for x in range(kMax + 1)][1:], 
                [(1 / x) * (2 ** n + (2 ** math.floor(n / x)) * (max(x - 2 ** (n % x), 1))) for x in range(kMax + 1)[1:]])
    plt.plot([x for x in range(kMax + 1)][1:], 
                [(1 / x) * (2 ** n + (2 ** math.floor(n / x)) * (x - 2 ** (n % x))) for x in range(kMax + 1)[1:]])


plotBoth(9, 9)