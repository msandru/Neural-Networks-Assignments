def get_determinant(m):
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if len(m) == 3:
        return (m[0][0] * m[1][1] * m[2][2] + \
                m[0][2] * m[1][0] * m[2][1] + \
                m[0][1] * m[1][2] * m[2][0]) - \
               (m[0][2] * m[1][1] * m[2][0] + \
                m[0][1] * m[1][0] * m[2][2] + \
                m[0][0] * m[1][2] * m[2][1])


def manual_method(x_coefficients, y_coefficients, z_coefficients, free):
    transpose = [x_coefficients] + [y_coefficients] + [z_coefficients]

    normal = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(0, 3):
        for j in range(0, 3):
            normal[i][j] = transpose[j][i]

    det = get_determinant(normal)

    if det == 0:
        print("No solution")
        exit(0)

    star_a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for i in range(0, 3):
        for j in range(0, 3):
            restricted_det = []
            for m in range(0, 3):
                restricted_line = []
                for n in range(0, 3):
                    if (n != j and m != i):
                        restricted_line.append(transpose[m][n])
                if len(restricted_line):
                    restricted_det.append(restricted_line)
            if ((i + j) % 2):
                det_restricted = -1 * get_determinant(restricted_det)
            else:
                det_restricted = get_determinant(restricted_det)
            star_a[i][j] = det_restricted

    invert = []
    for i in range(0, len(star_a)):
        line_invert = []
        for j in star_a[i]:
            line_invert.append(j / det)
        invert += [line_invert]

    numbers = []
    for i in range(0, 3):
        sum = 0
        for j in range(0, len(invert[i])):
            sum += invert[i][j] * free[j]
        numbers.append(sum)

    print("[manual method] The results for (x, y, z) are ", numbers)
