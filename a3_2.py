import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import randint


def parta(inputList):
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
    plt.figure('eCDF', figsize=(10, 5))
    plt.step(X, Y, label='eCDF')
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title('eCDF with %d samples. Sample mean = %.2f.' % (n, np.mean(X)))
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # seed random number generator
    seed(1)
    input = []
    for _ in range(10):
        input.append(randint(1, 99))
    parta(input)

    input = []
    for _ in range(100):
        input.append(randint(1, 99))
    parta(input)

    input = []
    for _ in range(1000):
        input.append(randint(1, 99))
    parta(input)