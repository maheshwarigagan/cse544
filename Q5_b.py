import numpy as np
import pandas as pd

if __name__ == '__main__':
    X = pd.read_csv('q5.csv', header=None)
    p_s = [0.1, 0.3, 0.5, 0.8]
    mu = 0.5
    sigma_2 = 1.0
    for index, row in X.iterrows():
        for p in p_s:
            prediction = []
            for w in row:
                current = (sigma_2*np.log((1-p)/p))/2*mu;
                if w >= current:
                    prediction.append(0)
                else:
                    prediction.append(1)
            print("For P(H_0) = ", p, ", the hypotheses selected are :: ", prediction)