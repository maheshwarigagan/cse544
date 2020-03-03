import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
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

def parte(showGraph=True):
    # Change the file path to q2.csv
    df = pd.read_csv('C:/Users/gagan/Downloads/q2.csv', header=None);
    df = df.values
    df = df.flatten()
    point, cdf = parta(df,False)
    z = 1.96
    upper_bound = []
    lower_bound = []
    n = len(point)
    for val in cdf:
        if val>1:
            val = 1.0
        upper_bound.append(val+z*(math.sqrt(val*(1-val)/n)))
        lower_bound.append(val-z*(math.sqrt(val*(1-val)/n)))

    X = point
    Y = cdf
    if showGraph:
        plt.figure('eCDF', figsize=(10, 5))
        plt.step(X, Y, label='eCDF')
        plt.fill_between(X, lower_bound, upper_bound, color='royalblue', alpha=0.5,
                         label='Normal Based 95% CI')
        plt.scatter(X, [0] * len(X), color='red', marker='x', s=100, label='samples')
        plt.xlabel('x')
        plt.ylabel('Pr[X<=x]')
        plt.title('eCDF with %d samples. Sample mean = %.2f.' % (n, np.mean(X)))
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()
    return lower_bound,upper_bound

def partf():
    # Change the file path to q2.csv
    df = pd.read_csv('C:/Users/gagan/Downloads/q2.csv', header=None);
    df = df.values
    df = df.flatten()
    point, cdf = parta(df,False)
    lower_bound,upper_bound = parte(showGraph=False)
    dkw_upper = []
    dkw_lower = []

    n = len(point)
    alpha = 0.05
    beta = math.sqrt((math.log(2 / alpha)) / (2 * n))
    for val in cdf:
        if val>1:
            val = 1.0
        dkw_lower.append(val-beta)
        dkw_upper.append(val+beta)
    X = point
    Y = cdf
    plt.figure('eCDF', figsize=(10, 5))
    plt.step(X, Y, color='yellow', label='F hat')
    plt.fill_between(X, lower_bound, upper_bound, color='blue', alpha=0.5,
                     label='Normal Based 95% CI')
    plt.fill_between(X, dkw_lower, dkw_upper, color='grey', alpha=0.7,
                     label='DKW Based 95% CI')

    plt.scatter(X, [0] * len(X), color='red', marker='x', s=100, label='samples')
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title('eCDF with %d samples. Sample mean = %.2f.' % (n, np.mean(X)))
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # seed random number generator
    seed(1)
    callPartA(10)
    callPartA(100)
    callPartA(1000)
    callPartC(10)
    callPartC(100)
    callPartC(1000)
    parte()
    partf()