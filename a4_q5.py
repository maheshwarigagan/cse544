import pandas as pd
import math


def calc_sample_mean(data):
    n = len(data)
    sum = 0
    for i in data:
        sum += i
    return sum / n


def calc_square_sum(data):
    square_sum = 0
    for i in data:
        square_sum += i * i
    return square_sum


def calc_second_moment(data):
    return calc_square_sum(data) / len(data)


# For normal, we have to estimate both the mu and sigma square
# formulas derived in lectures
def calc_normal_mme(data):
    u_hat = calc_sample_mean(data)
    sigma_square_hat = calc_second_moment(data) - (u_hat * u_hat)
    return round(u_hat, 3), round(sigma_square_hat, 3)


# For uniform, we have to find estimate for a and b
# formulas in assignment 4 question one part b
def calc_uniform_mme(data):
    # using the formula derived in 1b part of assignment 4
    # order of return is a_hat, b_hat
    # where a_hat is mean - root(3)*
    sample_mean = calc_sample_mean(data)
    sample_second_moment = calc_second_moment(data)
    return round(sample_mean - math.sqrt(3 * (sample_second_moment - sample_mean * sample_mean)), 3) \
        , round(sample_mean + math.sqrt(3 * (sample_second_moment - sample_mean * sample_mean)), 3)


# For exponential, we have to find estimate of lambda
# from result derived in 5a
def calc_exponential_mme(data):
    return round(1 / calc_sample_mean(data), 3)


# For normal, we have to estimate both the mu and sigma square
# formulas derived in lecture
def calc_normal_mle(data):
    u_hat = calc_sample_mean(data)
    sigma_hat = 0
    for i in data:
        sigma_hat += (i - u_hat) * (i - u_hat)
    sigma_hat = sigma_hat / len(data)
    return round(u_hat, 3), round(sigma_hat, 3)


# For uniform, we have to find estimate for a and b
# from derivation in lecture
def calc_uniform_mle(data):
    a_hat = min(data)
    b_hat = max(data)
    return a_hat, b_hat


# For exponential, we have to find estimate of lambda
# as derived in 5b
def calc_exponential_mle(data):
    return round(1 / calc_sample_mean(data), 3)


if __name__ == '__main__':
    # change file path to your local path
    acceleration_normal = pd.read_csv(
        "C:/Users/gagan/stonyBrook/probStats/code/ProbabilityAndStatistics/acceleration_normal.csv", header=None)
    acceleration_normal = acceleration_normal.values
    acceleration_normal = acceleration_normal.flatten()

    # change file path to your local path
    model_uniform = pd.read_csv(
        "C:/Users/gagan/stonyBrook/probStats/code/ProbabilityAndStatistics/model_uniform.csv", header=None)
    model_uniform = model_uniform.values
    model_uniform = model_uniform.flatten()

    # change file path to your local path
    mpg_exponential = pd.read_csv(
        "C:/Users/gagan/stonyBrook/probStats/code/ProbabilityAndStatistics/mpg_exponential.csv", header=None)
    mpg_exponential = mpg_exponential.values
    mpg_exponential = mpg_exponential.flatten()

    print("Question 5c")
    print("MME estimate for acceleration_normal data (u_hat, variance) is ", calc_normal_mme(acceleration_normal))
    print("MME estimate for model_uniform data(a_hat, b_hat) is ", calc_uniform_mme(model_uniform))
    print("MME estimate for mpg_exponential lambda_hat is ", calc_exponential_mme(mpg_exponential))

    print("Question 5d")
    print("MLE estimate for acceleration_normal data (u_hat, variance) is ", calc_normal_mle(acceleration_normal))
    print("MLE estimate for model_uniform data(a_hat, b_hat) is ", calc_uniform_mle(model_uniform))
    print("MLE estimate for mpg_exponential lambda_hat is ", calc_exponential_mle(mpg_exponential))
