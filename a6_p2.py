# imports
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.pyplot import figure

figure(figsize=(20, 20))


# returns (x, y^2) not y
def calculate_posterior(data, a, b2, sigma):
	mean = np.sum(data) / len(data)
	if mean != np.mean(data):
		print("Error in mean, Remove if statement after verification")
	se2 = (sigma * sigma) / len(data)
	x = ((b2 * mean) + (se2 * a)) / (b2 + se2)
	y2 = (b2 * se2) / (b2 + se2)
	return (x, y2)


# Expected the dataframe and take sigma default as 3
def calculate_plot(dataframe, sigma=3, prior=(0, 1)):
	mean_std_table = [prior]
	for index, values in dataframe.iterrows():
		posterior = calculate_posterior(values, prior[0], prior[1], sigma)
		print("After iteration ", index, ": Prior: ", prior, " posterior", posterior)
		mean_std_table.append(posterior)
		prior = posterior
		x = np.linspace(prior[0] - 3 * math.sqrt(prior[1]), prior[0] + 3 * math.sqrt(prior[1]), 5000)
		y = stats.norm.pdf(x, loc=prior[0], scale=math.sqrt(prior[1]))
		plt.plot(x, y, label="({u}, {s2})".format(u=prior[0], s2=prior[1]))
	plt.legend(loc='best')
	plt.show()

	print("Mean", "\t\t", "Variance")
	for row in mean_std_table:
		mean, std = row
		variance = math.sqrt(std)
		print(mean, "\t\t", variance)


if __name__ == '__main__':
	# read files
	# Part A
	print("Part A")
	sigma3 = pd.read_csv("q2_sigma3.dat", header=None)
	calculate_plot(sigma3)

	# Part B
	print("Part B")
	sigma100 = pd.read_csv("q2_sigma100.dat", header=None)
	calculate_plot(sigma100, sigma=100)
