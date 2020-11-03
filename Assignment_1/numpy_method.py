import numpy as np


def numpy_method(x_coefficients, y_coefficients, z_coefficients, free):
    transpose_matrix = [x_coefficients] + [y_coefficients] + [z_coefficients]

    normal = np.transpose(transpose_matrix)

    det = np.linalg.det(normal)
    if det == 0:
        print("No solution")
        exit(0)

    inverse_matrix = np.linalg.inv(normal)

    result = np.dot(inverse_matrix, free)

    print("[numpy method] The results for (x, y, z) are ", result)
