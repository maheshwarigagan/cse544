import numpy as np


# Function to find the steady state matrix for the transition matrix created.
def steady_state_power(transition_matrix: np.ndarray):
    exp = 40
    mat1 = np.linalg.matrix_power(transition_matrix, exp)
    mat2 = np.linalg.matrix_power(transition_matrix, exp + 1)
    while not np.array_equal(mat1, mat2):
        exp *= 2
        mat1 = np.linalg.matrix_power(transition_matrix, exp)
        mat2 = np.linalg.matrix_power(transition_matrix, exp + 1)
    return mat1[0, :]


if __name__ == '__main__':
    x = [[0.9, 0, 0.1, 0], [0.8, 0, 0.2, 0], [0, 0.5, 0, 0.5], [0, 0.1, 0, 0.9]]
    steady_state = steady_state_power(np.array(x))
    print("Steady_State: Power iteration in format[CC, CS, SC, SS]>>", steady_state)
    Pr_s = np.array([[0.1], [0.2], [0.5], [0.9]])
    solution = np.matmul(steady_state, Pr_s)
    print("Numerical solution for part B as product of: \n",steady_state, " and \n", Pr_s, " Is: ", solution)

# def steady_state_power(transition_matrix):
#     power = 2
#     while (True):
#         transition_matrix = np.linalg.matrix_power(transition_matrix, power)
#         elements_equal = True
#         # We have to change this logic of element comparison of rows, help needed!!!
#         for i in range(len(transition_matrix[0])):
#             if abs(transition_matrix[0][i] - transition_matrix[1][i]) > 10 ** -5:
#                 elements_equal = False
#         print(power)
#         print(transition_matrix[0])
#         if elements_equal:
#             break
#         power = power + 1
#
#     return power,transition_matrix[0]
#
#
# if __name__ == '__main__':
#     print(steady_state_power(np.array([[0.9, 0.1, 0, 0], [0, 0, 0.8, 0.2], [0.5, 0.5, 0, 0], [0, 0, 0.1, 0.9]])))
