import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ks_test(X, Y):
    X = np.sort(X)
    Y = np.sort(Y)
    max_diff = float("-inf")
    max_x = 0
    y_min = 0
    y_max = 0

    for y in Y:
        count = 0;
        for x in X:
            if x < y:
                count = count + 1
        f_hat_x = count / len(X)
        f_hat_y_neg = np.searchsorted(Y, y, side='left') / len(Y)
        f_hat_y_pos = np.searchsorted(Y, y, side='right') / len(Y)
        if abs(f_hat_x - f_hat_y_neg) > max_diff or abs(f_hat_x - f_hat_y_pos) > max_diff:
            max_diff = max(max_diff, abs(f_hat_x - f_hat_y_neg), abs(f_hat_x - f_hat_y_pos))
            max_x = y
            y_min = max(y_min, f_hat_y_neg)
            y_max = max(y_max, f_hat_x)

    return max_x, max_diff, y_min, y_max


def permutation_test(X, Y, p_count, p_threshold):
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    t_obs = abs(x_bar - y_bar)
    C = np.concatenate((X, Y))
    count = 0
    for i in range(p_count):
        p = np.random.permutation(C)
        mid = int(len(C) / 2)
        t_i = abs(np.mean(p[:mid]) - np.mean(p[mid:]))
        if t_i > t_obs:
            count += 1
    p_value = count / p_count
    print("p_value is: ", p_value)
    if p_value <= p_threshold:
        print("Rejecting Null Hypothesis, X and Y don't come from same distribution")
    else:
        print("Accepting Null Hypothesis, X and Y come from same distribution")


# code from assignment 3
def calculate_ecdf(S):
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
    X = np.sort(X)

    Y = pd.read_csv('2009.csv', header=None)
    Y = Y.values
    Y = Y.flatten()
    Y = np.sort(Y)

    Z = pd.read_csv('2019.csv', header=None)
    Z = Z.values
    Z = Z.flatten()
    Z = np.sort(Z)

    print("4a: permutation test with n = 200")
    print("Comparing 1999 and 2009")
    permutation_test(X, Y, 200, 0.05)
    print("Comparing 2009 and 2019")
    permutation_test(Y, Z, 200, 0.05)

    print("")
    print("4b: permutation test with n=2000")
    print("Comparing 1999 and 2009")
    permutation_test(X, Y, 2000, 0.05)
    print("Comparing 2009 and 2019")
    permutation_test(Y, Z, 2000, 0.05)

    print("")
    print("4c")
    print("comparing data for 1999 and 2009")
    x_axis, y_axis, sorted_points = calculate_ecdf(X)
    drawplot(x_axis, y_axis, sorted_points, label="1999", marker='x', color='red')

    x_axis, y_axis, sorted_points = calculate_ecdf(Y)
    drawplot(x_axis, y_axis, sorted_points, label="2009", marker='o', color='blue')

    score, max_diff, min_cdf, max_cdf = ks_test(X, Y)
    print("At x= ", score, "we have the maximum difference(D value) of: ", max_diff, "corresponding minimum value of cdf: ", min_cdf, "and maximum value of cdf: ", max_cdf)
    if max_diff > 0.05:
        print("D value is greater than threshold, rejecting null hypothesis: X and Y don't come from same distribution")
    else:
        print("D value is less than threshold, accepting null hypothesis: X and Y come from same distribution")
    plt.plot([score, score], [min_cdf, max_cdf])
    plt.annotate(round(max_diff, 3), (score + 1.5, 0.5), textcoords="offset points", xytext=(0, 15), ha='center')

    plt.show()

    print("comparing data for 2009 and 2019")
    x_axis, y_axis, sorted_points = calculate_ecdf(Y)
    drawplot(x_axis, y_axis, sorted_points, label="2009", marker='o', color='blue')

    x_axis, y_axis, sorted_points = calculate_ecdf(Z)
    drawplot(x_axis, y_axis, sorted_points, label="2019", marker='^', color='green')

    score, max_diff, min_cdf, max_cdf = ks_test(Y, Z)
    print("At x= ", score, "we have the maximum difference(D value) of: ", max_diff, "corresponding minimum value of cdf: ", min_cdf, "and maximum value of cdf: ", max_cdf)
    if max_diff > 0.05:
        print("D value is greater than threshold, rejecting null hypothesis: X and Y don't come from same distribution")
    else:
        print("D value is less than threshold, accepting null hypothesis: X and Y come from same distribution")

    plt.plot([score, score], [min_cdf, max_cdf])
    plt.annotate(round(max_diff, 3), (score + 1.5, 0.5), textcoords="offset points", xytext=(0, 15), ha='center')

    plt.plot([score, score], [min_cdf, max_cdf])
    plt.annotate(round(max_diff, 3), (score + 1.5, 0.5), textcoords="offset points", xytext=(0, 15), ha='center')

    plt.show()
