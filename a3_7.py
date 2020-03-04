import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from operator import add
import scipy.stats as ss


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

    # print(data)
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


# https://stackoverflow.com/questions/49106806/how-to-do-a-simple-gaussian-mixture-sampling-and-pdf-plotting-with-numpy-scipy
def gausian_mixture(alphas=np.zeros(800)):
    n = 10000
    np.random.seed(0x5eed)
    # Parameters of the mixture components
    norm_params = np.array([[0, 1],
                            [3, 1],
                            [6, 1],
                            [9, 1]])
    n_components = norm_params.shape[0]
    # Weight of each component, in this case all of them are 1/3
    weights = np.ones(n_components, dtype=np.float64) / 4.0
    # A stream of indices from which to choose the component
    mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
    # y is the mixture sample
    y = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                    dtype=np.float64)

    # Theoretical PDF plotting -- generate the x and y plotting positions
    xs = alphas
    ys = np.zeros_like(xs)

    for (l, s), w in zip(norm_params, weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w

    return ys


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
    ax.plot(alphas, gausian_mixture(alphas), 'r-', lw=5, alpha=0.6, label='True')
    ax.plot(alphas, kdes[0], 'g-', lw=5, alpha=0.6, label='h=0.1')
    ax.plot(alphas, kdes[1], 'b-', lw=5, alpha=0.6, label='h=1')
    ax.plot(alphas, kdes[2], 'y-', lw=5, alpha=0.6, label='h=7')
    print("Check plot")
    plt.legend(loc='best')
    plt.show()


def part_b():
    dataset = []
    h = [0.01, .1, .3, .6, 1, 3, 7]
    alphas = np.arange(-5, 10, 0.1)
    true_d = gausian_mixture(alphas)
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
        expectation = np.true_divide(expectation, 150)
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
    # dummy value
    val = 10000000000000000
    key = -1
    for (k, v) in bias2_tot.items():
        sum = v + variance_tot[k]
        if sum < val:
            val = sum
            key = k

    print("The Optimal value of h you should use is ", key)


if __name__ == '__main__':
    plt.figure(figsize=(20, 10))
    print("Starting part A")
    part_a()
    print("Starting part B")
    part_b()
