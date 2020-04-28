import numpy as np
import pandas as pd

# SIC: Signature given the the assignment
def MAP_descision(w, h_0):
    sum_of_data = np.sum(w)
    current = (sigma_2 * np.log(h_0/(1 - h_0))) / (2 * mu)
    if sum_of_data >= current:
        return 0
    else:
        return 1


if __name__ == '__main__':
    X = pd.read_csv('q5.csv', header=None)
    p_s = [0.1, 0.3, 0.5, 0.8]
    mu = 0.5
    sigma_2 = 1.0
    for p in p_s:
        prediction = []
        for (columnName, columnData) in X.iteritems():
            prediction.append(MAP_descision(columnData, p))
        print("For P(H_0) = ", p, ", the hypotheses selected are :: ", prediction)
