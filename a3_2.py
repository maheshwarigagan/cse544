import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import randint


def parta(inputList,showGraph=True):
    n = len(inputList)
    probabilities = dict()
    inputList.sort()
    for entry in inputList:
        if entry not in probabilities:
            probabilities[entry] = 1.0
        else:
            probabilities[entry] += 1.0
    for key in probabilities.keys():
        probabilities[key] = probabilities[key] / n
    cdf = []
    cumulative = 0
    cdf.append(0)
    for value in probabilities.values():
        cumulative = cumulative + value
        cdf.append(cumulative)
    X = [0] + list(probabilities.keys())
    Y = cdf
    if showGraph:
        plt.figure('eCDF', figsize=(10, 5))
        plt.step(X, Y, label='eCDF')
        plt.scatter(X, [0] * len(X), color='red', marker='x', s=100, label='samples')
        plt.xlabel('x')
        plt.ylabel('Pr[X<=x]')
        plt.title('eCDF with %d samples. Sample mean = %.2f.' % (n, np.mean(X)))
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()
    return X[1:], cdf[1:]

def partc(inputList):
    n = len(inputList)
    cdfList = np.zeros((n,99))
    for i in range(0, n):
        point, cdf = parta(inputList[i],False)

        for j in range(len(point)):
            cdfList[i][point[j]-1] = cdf[j]

        prev = 0
        for j in range(99):
            if cdfList[i][j] == 0:
                cdfList[i][j] = prev
            else:
                prev = cdfList[i][j]

    X = np.arange(0,99,1)
    Y = cdfList.mean(0)
    plt.figure('eCDF', figsize=(10, 5))
    plt.step(X, Y, label='eCDF')
    plt.scatter(X, [0] * 99, color='red', marker='x', s=100, label='samples')
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title('eCDF with %d samples. Sample mean = %.2f.' % (n, np.mean(X)))
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()



def callPartA(times):
    input = []
    for _ in range(times):
        input.append(randint(1, 99))
    parta(input)

def callPartC(times):
    l = list()
    for _ in range(times):
        t = []
        for _ in range(10):
            t.append(randint(1, 99))
        l.append(t)
    partc(l)


if __name__ == '__main__':
    # seed random number generator
    seed(1)
    callPartA(10)
    callPartA(100)
    callPartA(1000)
    callPartC(10)
    callPartC(100)
    callPartC(1000)