import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from operator import add


def create_dataset():
    data = []
    x_unif = np.random.uniform(0, 1, 800)
    for val in x_unif:
        x = -1
        if val <= 0.25:
            x = np.random.normal(0, 1)
        elif 0.25 < val <= 0.5:
            x = np.random.normal(3, 1)
        elif 0.5 < val <= 0.75:
            x = np.random.normal(6, 1)
        else:
            x = np.random.normal(9, 1)
        data.append(x)

    print(data)
    return data


def kde_estimates(data, h):
    alpha = np.arange(-5, 10, 0.1)
    values = []
    for a in alpha:
        sum = 0
        for x in data:
            val = abs(a - x)
            if val <= h / 2:
                sum += 1
        nh = h * len(data)
        sum = sum / nh
        values.append(sum)
    return values


def part_a():
    print("Creating dataset")
    data = create_dataset()
    h = [0.1, 1, 7]
    kdes = []
    alphas = np.arange(-5, 10, 0.1)
    print("Calculating estimates will take a while")
    for hi in h:
        print("Calculating estimates for h = ", hi)
        v = kde_estimates(data, hi)
        kdes.append(v)
    fig, ax = plt.subplots(1, 1)
    ax.plot(alphas, get_alpha_distribution(alphas), 'r-', lw=5, alpha=0.6, label='True')
    ax.plot(alphas, kdes[0], 'g-', lw=5, alpha=0.6, label='h=0.1')
    ax.plot(alphas, kdes[1], 'b-', lw=5, alpha=0.6, label='h=1')
    ax.plot(alphas, kdes[2], 'y-', lw=5, alpha=0.6, label='h=7')
    print("Check plot")
    plt.legend(loc='best')
    plt.show()


def get_alpha_distribution(alphas):
    true_d = []
    x_unif = np.random.uniform(0, 1, len(alphas))
    for val, alp in zip(x_unif, alphas):
        x = -1
        if val <= 0.25:
            x = norm.pdf(alp, 0, 1)
        elif 0.25 < val <= 0.5:
            x = norm.pdf(alp, 3, 1)
        elif 0.5 < val <= 0.75:
            x = norm.pdf(alp, 6, 1)
        else:
            x = norm.pdf(alp, 9, 1)
        true_d.append(x)
    return true_d

def part_b():
    dataset = []
    h = [0.01, .1, .3, .6, 1, 3, 7]
    alphas = np.arange(-5, 10, 0.1)
    true_d = get_alpha_distribution(alphas)
    bias2_tot = dict()
    variance_tot = dict()
    for i in range(0, 151):
        data = create_dataset()
        dataset.append(data)

    for hi in h:
        kdes = []
        expectation = [0] * 800
        print("Calculating estimates for h = ", hi)
        for data in dataset:
            v = kde_estimates(data, hi)
            kdes.append(v)
            expectation = list(map(add, expectation, v))
        div = np.repeat(150, 150)
        expectation = np.array(expectation) / np.array(div)
        sum_o_squares = np.repeat(0, 150)
        for kde in kdes:
            arr = np.array(kde)
            arr = arr - expectation
            arr = arr * arr
            sum_o_squares = sum_o_squares + arr
        variance = sum_o_squares / div
        bias = expectation - true_d
        bias2 = bias * bias
        bias2_tot[hi] = np.sum(bias2) / len(alphas)
        variance_tot[hi] = np.sum(variance) / len(alphas)
    # print(bias2_tot)
    # print(variance_tot)
    print("Calculations complete")
    biases = sorted(bias2_tot.items())
    fig, (b, v) = plt.subplots(2)
    x, y = zip(*biases)
    b.plot(x, y, 'r--', label="bias")
    b.set_title("Bias vs h")
    variances = sorted(variance_tot.items())
    a, b = zip(*variances)
    v.plot(a, b, "b--", label="variance")
    v.set_title("Variance vs h")
    print("Check figure")
    plt.show()
    #b2

    print("Optimal value for h is", np.argmax(np.array(y) + np.array(b)))


if __name__ == '__main__':
    plt.figure(figsize=(20, 10))
    print("Starting part A")
    part_a()
    print("Starting part B")
    part_b()
