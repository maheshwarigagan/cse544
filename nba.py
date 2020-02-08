import numpy as np
from operator import add

np.random.seed(960)

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
def complex_experiment(size=10):
    # Simulating the first game of the series and then moving on with the rest of the games
    # As given in Piazza post 30 we need to assume that the first two games were held in toroto
    # And the next two games were held in philli
    # TOR home games
    game1 = np.random.binomial(n=1, p=0.75, size=size)
    game2 = np.random.binomial(n=1, p=0.75, size=size)

    # Philli Home games
    game3 = np.random.binomial(n=1, p=0.25, size=size)
    game4 = np.random.binomial(n=1, p=0.25, size=size)

    values_after_4_games = list(map(add, map(add, map(add, game1, game2), game3), game4))
    no_of_draws = list(values_after_4_games).count(2)
    # print(no_of_draws)
    # We only need the games in which the value is exactly 2. Those are the games which were tied.
    game5 = np.random.binomial(n=1, p=0.75, size=no_of_draws)
    game6 = np.random.binomial(n=1, p=0.25, size=no_of_draws)

    # Once again we just need to look at the values where the games were tied 3-3 in the season which
    # is the score of 1 in case of the sum of game 5 and game 6 score
    score_after_game_5_and_6 = map(add, game5, game6)
    no_of_draws_after_game_6 = list(score_after_game_5_and_6).count(1)
    # print(no_of_draws_after_game_6)

    game7 = np.random.binomial(n=1, p=0.75, size=no_of_draws_after_game_6)
    no_of_cases_where_tor_won = list(game7).count(1)
    # print(no_of_cases_where_tor_won)
    return no_of_cases_where_tor_won / no_of_draws


print(complex_experiment(size=pow(10, 6)))


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
    #
    # #     print("For n = ", x, "No Of Draws = ", count, "Probability = ", prob)

    print("")
    # game and run a simulation
    # for x in range(3, 8):
    prob = complex_experiment(size=pow(10, N))
    print("For N = ", N, ", the simulated value for part (e) is ", prob)
    # print("For n = ", x, "No Of favorable outcomes = ", tor_win, "Total no of ties after 4 games = ",
    #       tie_count, "Probability = ", prob)


#
#
# # main(N=n)
main(N=3)
print("--")
main(N=4)
print("--")
main(N=5)
print("--")
main(N=6)
print("--")
main(N=7)
