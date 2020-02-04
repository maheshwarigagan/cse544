import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

np.random.seed(640)


# Fair 4 games probability
def probability(events, event):
    count = 0
    for ele in events:
        if (ele == event):
            count = count + 1
    # print(event, ":", count)
    total_events = len(events)
    return count, count / total_events


# Default values are fair game value of n is 4 and the size of the experiment is 100
def experiment(n=4, p=0.5, size=pow(10, 2), target=2):
    val = np.random.binomial(n=n, p=p, size=size)
    return probability(val, target)


# Part F Subparts
# Part (a)
# for n in range(3, 8):
#     count, prob = experiment(size=pow(10, n))
#     print("For n = ", n, "No Of Draws = ", count, "Probability = ", prob)

# Part(c) The target is now 4 and total number of instances are 7
# for x in range(3, 8):
#     count, prob = experiment(n=7, size=pow(10, x), target=4)
#     print("For n = ", x, "No Of Draws = ", count, "Probability = ", prob)

# Part e
# for x in range(3, 8):
#     count, prob = experiment(n=7, size=pow(10, x), target=4)
#     print("For n = ", x, "No Of Draws = ", count, "Probability = ", prob)
