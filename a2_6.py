import numpy as np

np.random.seed(960)


def rand_walk(init_val, final_high, final_low, prob, N):
    total_low = 0
    total_high = 0
    total_steps = 0
    total_expected = 0

    for i in range(N):
        current = init_val
        steps = 0
        while current != final_high and current != final_low:
            p = np.random.uniform(0, 1)
            if (p < prob):
                current += 1
            else:
                current -= 1
            steps += 1

        if current == final_high:
            total_high += 1
        else:
            total_low += 1
        total_expected += current
        total_steps += steps

    a = total_high / N
    b = total_low / N
    c = total_expected / N
    d = total_steps / N
    return a, b, c, d


if __name__ == '__main__':
    print(rand_walk(100, 150, 0, 0.5, 10000))
    print(rand_walk(100, 200, 0, 0.52, 10000))
    print(rand_walk(200, 250, 0, 0.54, 10000))
