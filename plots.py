from subset import *

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
    #plt.plot([x for x in range(kMax + 1)][1:], 
    #            [(1 / x) * (2 ** n + (2 ** math.floor(n / x)) * (max(x - 2 ** (n % x), 1))) for x in range(kMax + 1)[1:]])
    plt.plot([x for x in range(kMax + 1)][1:], 
                [(1 / x) * (2 ** n + (2 ** math.floor(n / x)) * (x - 2 ** (n % x))) for x in range(kMax + 1)[1:]])