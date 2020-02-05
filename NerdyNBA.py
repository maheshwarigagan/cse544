import numpy as np
from operator import add

np.random.seed(640)

# README: Main function at the bottom of the file; Change the value for n to get other results.
# README: Call main function multiple times to view results together.
# n = 3
n = 4


# n=5
# n=6
# n=7

# Fair 4 games probability
def probability(events, event):
    count = 0
    for ele in events:
        if ele == event:
            count = count + 1
    # print(event, ":", count)
    total_events = len(events)
    return count, count / total_events


# Default values are fair game value of n is 4 and the size of the experiment is 100
def experiment(n=4, p=0.5, size=pow(10, 2), target=2):
    val = np.random.binomial(n=n, p=p, size=size)
    return probability(val, target)


def custom_probability(events, event):
    # N_a
    tie_count = 0
    # N_ba
    tor_win = 0
    for ele in events:
        if ele == event:
            tie_count += 1
            val = np.random.binomial(n=1, p=0.75)
            if val == 1:
                tor_win += 1
                # Prob(B|A) = N_ba/N_a
    return tor_win, tie_count, tor_win / tie_count


# Complex experiment
def complex_experiment(size=10, target=2):
    val1 = np.random.binomial(n=1, p=0.75, size=size)
    val2 = np.random.binomial(n=1, p=0.25, size=size)
    val = map(add, val1, val2)
    result = list(val)
    return custom_probability(result, 1)


def main(N=3):
    # Part (a)
    # for n in range(3, 8):
    count, prob = experiment(size=pow(10, N))
    print("For N = ", N, ", the simulated value for part (a) is ", prob)
    # print("For n = ", n, "No Of Draws = ", count, "Probability = ", prob)

    print("")
    # Part(c) The target is now 4 and total number of instances are 7
    # for x in range(3, 8):
    count, prob = experiment(n=4, size=pow(10, N), target=2)
    print("For N = ", N, ", the simulated value for part (c) is ", prob)

    #     print("For n = ", x, "No Of Draws = ", count, "Probability = ", prob)

    print("")
    # game and run a simulation
    # for x in range(3, 8):
    tor_win, tie_count, prob = complex_experiment(size=pow(10, N), target=2)
    print("For N = ", N, ", the simulated value for part (e) is ", prob)
    # print("For n = ", x, "No Of favorable outcomes = ", tor_win, "Total no of ties after 4 games = ",
    #       tie_count, "Probability = ", prob)


# main(N=n)
main(N=3)
print("--")
main(N=4)
print("--")
main(N=5)
print("--")
main(N=6)
print("--")
main(N=7)
