import numpy as np


def steady_state_power(transition_matrix):
    power = 2
    while (True):
        transition_matrix = np.linalg.matrix_power(transition_matrix, power)
        elements_equal = True
        # We have to change this logic of element comparison of rows, help needed!!!
        for i in range(len(transition_matrix[0])):
            if abs(transition_matrix[0][i] - transition_matrix[1][i]) > 10 ** -5:
                elements_equal = False
        print(power)
        print(transition_matrix[0])
        if elements_equal:
            break
        power = power + 1

    return power,transition_matrix[0]


if __name__ == '__main__':
    print(steady_state_power(np.array([[0.9, 0.1, 0, 0], [0, 0, 0.8, 0.2], [0.5, 0.5, 0, 0], [0, 0, 0.1, 0.9]])))
