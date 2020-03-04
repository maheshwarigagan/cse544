import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def partc(showGraph = True):
    # Change the file path to weather.csv
    df = pd.read_csv('C:/Users/gagan/Downloads/weather.csv', header=None);
    df = df.values
    df = df.flatten()
    n = len(df)
    lowest_bin = math.floor(min(df))
    highest_bin = math.ceil(max(df))

    countPerBin = dict()
    for val in df:
        floored_val = math.floor(val)
        if floored_val not in countPerBin:
            countPerBin[floored_val] = 1.0
        else:
            countPerBin[floored_val] += 1.0

    graph_data = []
    for val in range(lowest_bin,highest_bin):
        if val not in countPerBin:
            graph_data.append(0)
        else:
            graph_data.append(countPerBin.get(val))

    graph_data = [x / n for x in graph_data]

    if showGraph:
        plt.title('histogram estimate')
        plt.xlabel('x')
        plt.ylabel('h[x]')
        plt.xticks(np.arange(lowest_bin,highest_bin,1))
        plt.yticks(np.arange(0, 1, 0.01))
        plt.bar(np.arange(lowest_bin,highest_bin,1),graph_data)

        plt.show()
    return list(np.arange(lowest_bin,highest_bin,1)),graph_data

def partd():
    x,pdf = partc(showGraph=False)
    X = [0] + x
    cdf = []
    cumulative = 0
    cdf.append(0)

    for value in pdf:
        cumulative = cumulative + value
        cdf.append(cumulative)

    Y = cdf
    X = X[1:]
    Y = Y[1:]
    plt.figure('eCDF', figsize=(10, 5))
    plt.xticks(np.arange(X[0], X[len(X)-1]+1, 1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.step(X, Y, label='eCDF')
    plt.scatter(X, [0] * len(X), color='red', marker='x', s=100, label='samples')
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title('eCDF')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    partc()
    partd()