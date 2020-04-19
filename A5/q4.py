import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def permutation_test(X, Y, p_count, p_threshold):
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    t_obs = abs(x_bar - y_bar)
    C = np.concatenate((X, Y))
    n = len(X) + len(Y)
    count = 0
    for i in range(p_count):
        p = np.random.permutation(C)
        mid = int(len(C) / 2)
        t_i = abs(np.mean(p[:mid]) - np.mean(p[mid:]))
        if t_i > t_obs:
            count += 1
    print("count: ", count)
    p_value = count / p_count
    if p_value <= p_threshold:
        print("Rejecting Null Hypothesis, X and Y don't come from same distribution")
    else:
        print("Accepting Null Hypothesis, X and Y come from same distribution")


# code from assignment 3
def parta(S, showGraph=True):
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
    X.insert(0, 0)
    Y.insert(0, 0)

    # if showGraph:
    #     drawplot(X,Y,Z,S)
    # Returning from index one, since we inserted an extra entry of 0 to make our
    # graph adhere to the requirements.
    return X, Y, S


def drawplot(X, Y, S, label, marker='x', color='red'):
    plt.figure('eCDF', figsize=(15, 10))
    plt.step(X, Y, label='eCDF', color=color)
    plt.xticks(np.arange(0, max(X) + 1, 1), rotation=90)
    plt.scatter(S, [0] * len(S), color=color, marker=marker, s=100, label=label)
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title('eCDF')
    plt.legend(loc="upper left")
    plt.grid()
    # plt.show()


if __name__ == '__main__':
    X = pd.read_csv('1999.csv', header=None)
    X = X.values
    X = X.flatten()
    Y = pd.read_csv('2009.csv', header=None)
    Y = Y.values
    Y = Y.flatten()
    Z = pd.read_csv('2019.csv', header=None)
    Z = Z.values
    Z = Z.flatten()

    print("4a")
    permutation_test(X, Y, 200, 0.05)
    permutation_test(Y, Z, 200, 0.05)

    print("4b")
    permutation_test(X, Y, 2000, 0.05)
    permutation_test(Y, Z, 2000, 0.05)

    print("4c")
    x_axis, y_axis, sorted_points = parta(X, showGraph=False)
    # print("x_axis")
    # print(x_axis)
    # print("y_axis")
    # print(y_axis)
    # print("sorted_points")
    # print(sorted_points)
    # X = [3.6,2.1,1.8]
    # Y = [2.9,0.4]
    # for()



    drawplot(x_axis, y_axis, sorted_points, label="1999", marker='x', color='red')

    x_axis, y_axis, sorted_points = parta(Y, showGraph=False)
    drawplot(x_axis, y_axis, sorted_points, label="2009", marker='o', color='blue')

    x_axis, y_axis, sorted_points = parta(Z, showGraph=False)
    drawplot(x_axis, y_axis, sorted_points, label="2019",marker='^', color='green')

    data1 = np.sort(X)
    data2 = np.sort(Y)
    n1 = len(data1)
    n2 = len(data2)
    data_all = np.concatenate([data1,data2])
    cdf1 = np.searchsorted(data1, data_all, side='right')/(1.0*n1)
    print(np.searchsorted(data1, data_all, side='right'))
    cdf2 = (np.searchsorted(data2, data_all, side='right'))/(1.0*n2)
    print(np.searchsorted(data2, data_all, side='right'))
    d = np.max(np.absolute(cdf1-cdf2))
    print(d)


    plt.show()
