import math
import numpy as np

# README: Uncomment any test-case and run the file to execute the simulation.

(n, N) = (5, pow(10, 2))
# (n, N) = (5, pow(10, 3))
# (n, N) = (5, pow(10, 4))
# (n, N) = (5, pow(10, 5))


# (n, N) = (20, pow(10, 2))
# (n, N) = (20, pow(10, 3))
# (n, N) = (20, pow(10, 4))
# (n, N) = (20, pow(10, 5))

favorable_outcomes = 0
for i in range(1, N):
    iPhone_list = list(range(1, n + 1))
    np.random.shuffle(iPhone_list)

    keep = 0
    for index, val in enumerate(iPhone_list):
        if (index+1) == val:
            keep += 1

    if keep > 0:
        favorable_outcomes += 1

probability = favorable_outcomes / N
# print("Favorable outcomes: ", favorable_outcomes, ", Total number of outcomes: ", N)
print("For (n, N) = (", n, ",", N, ") the approximated value for probability is ", probability)
