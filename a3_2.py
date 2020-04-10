import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from random import seed
from random import randint

def drawplot(X,Y,S):
    plt.figure('eCDF', figsize=(15, 10))
    plt.step(X, Y, label='eCDF')
    plt.xticks(np.arange(0,max(X)+1,1), rotation=90)
    plt.scatter(S, [0] * len(S), color='red', marker='x', s=100, label='samples')
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title('eCDF')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

# Code for question 2 part a
def parta(S,showGraph=True):

    # From https://www3.cs.stonybrook.edu/~anshul/courses/cse544_s20/eCDF.py
    n = len(S)
    Srt = sorted(S)
    delta = 1
    X = [min(Srt) - delta]
    Y = [0]
    for i in range(0, n):
        X = X + [Srt[i], Srt[i]]
        Y = Y + [Y[len(Y) - 1], Y[len(Y) - 1] + (1 / n)]
    X = X + [max(Srt)]
    Y = Y + [1]

    # Insert a 0,0, since question asks for graph to start from zero
    X.insert(0,0)
    Y.insert(0,0)

    if showGraph:
        drawplot(X,Y,S)
    # Returning from index one, since we inserted an extra entry of 0 to make our
    # graph adhere to the requirements.
    return X[1:], Y[1:]

# Code for question 2 part c
def partc(inputList):
    n = len(inputList)
    cdfList = np.zeros((n,99))
    samplePoints = set()
    for i in range(0, n):
        point, cdf = parta(inputList[i],False)

        for j in range(len(point)):
            samplePoints.add(point[j])
            cdfList[i][point[j]-1] = cdf[j]

        prev = 0
        for j in range(99):
            if cdfList[i][j] == 0:
                cdfList[i][j] = prev
            else:
                prev = cdfList[i][j]

    X = np.arange(0,99,1)
    Y = cdfList.mean(0)
    drawplot(X,Y,list(samplePoints))

# Code for question 2 part b
def callPartA(times):
    input = []
    for _ in range(times):
        input.append(randint(1, 99))
    parta(input)

# Code for question 2 part d
def callPartC(times):
    l = list()
    for _ in range(times):
        t = []
        for _ in range(10):
            t.append(randint(1, 99))
        l.append(t)
    partc(l)

# Code for question 2 part e
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
        plt.figure('eCDF', figsize=(15, 10))
        plt.step(X, Y, color='green', label='eCDF')
        plt.step(X,lower_bound, color='blue', label='Normal Based 95% lower CI')
        plt.step(X,upper_bound, color='blue', label='Normal Based 95% lower CI')
        plt.xlim([0,2])
        plt.scatter(X, [0] * len(X), color='red', marker='x', s=100, label='samples')
        plt.xlabel('x')
        plt.ylabel('Pr[X<=x]')
        plt.title('eCDF')
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()
    return lower_bound,upper_bound

# Code for question 2 part f
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
    plt.figure('eCDF', figsize=(15, 10))
    plt.step(X, Y, color='yellow', label='F hat')
    plt.xlim([0, 2])
    plt.step(X, lower_bound, color='blue', label='Normal Based 95% lower CI')
    plt.step(X, upper_bound, color='blue', label='Normal Based 95% upper CI')
    plt.step(X, dkw_lower, color='grey', label='DKW Based 95% lower CI')
    plt.step(X, dkw_upper, color='grey', label='DKW Based 95% upper CI')

    plt.scatter(X, [0] * len(X), color='red', marker='x', s=100, label='samples')
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title('eCDF')
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