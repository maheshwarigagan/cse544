import numpy as np
from operator import add

np.random.seed(960)

# README: Main function at the bottom of the file; Change the value for n to get other results.
# README: Call main function multiple times to view results together.
N = 7


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
def experiment_a(n=4, p=0.5, size=pow(10, 2), target=2):
    val = np.random.binomial(n=n, p=p, size=size)
    return probability(val, target)



# Complex experiment
def experiment_e(size):
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


def experiment_c(size):
    # First 4 games only look for 2-2 score
    count = list(np.random.binomial(n=4, p=0.5, size=size)).count(
        2)
    # Next two games only look for 1-1 draw in the next two
    draws_after_33 = list(np.random.binomial(n=2, p=0.5, size=count)).count(1)

    # finally look for TOR win.
    final_4_3_wins = list(np.random.binomial(n=1, p=0.5, size=draws_after_33)).count(1)
    prob = final_4_3_wins / count
    return prob



if __name__ == '__main__':
    # Part(a)
    count, prob = experiment_a(size=pow(10, N))
    print("For N = ", N, ", the simulated value for part (a) is ", prob)

    print("")
    # Part(c)
    prob = experiment_c(size=pow(10,N))
    print("For N = ", N, ", the simulated value for part (c) is ", prob)

    print("")
    # Part(d)
    prob = experiment_e(size=pow(10, N))
    print("For N = ", N, ", the simulated value for part (e) is ", prob)
