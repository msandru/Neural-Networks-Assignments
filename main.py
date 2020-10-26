import numpy_method as numpy_method
import manual_method as manual_method

def get_coefficients(parameter_type, equations):
    parameters = []
    for i in range(0, len(equations)):

        equation = equations[i];

        if equation.count(parameter_type) != 0:
            sequence = equation.split(parameter_type, 1)[0]
            value = ""
            for i in range(1, len(sequence) + 1):
                value = str(sequence[-i]) + value
                if sequence[-i] in ["-", "+"]:
                    break
            if value == '+':
                parameters.append(1)
            elif value == '-':
                parameters.append(-1)
            elif value == '':
                parameters.append(1)
            else:
                parameters.append(int(value))

        else:
            parameters.append(0)

    return parameters


def main(file_name):
    file_content = open("equations", "r")
    text = file_content.read()
    equations = list((text).split("\n"))
    equations = list(equation.replace(" ", "") for equation in equations)
    print(equations)

    dict = {}

    dict["x"] = get_coefficients("x", equations)
    dict["y"] = get_coefficients("y", equations)
    dict["z"] = get_coefficients("z", equations)
    dict["free"] = list(int(equation.rsplit("=", 1)[1]) for equation in equations)

    print(dict["x"], dict["y"], dict["z"], dict["free"])

    numpy_method.numpy_method(dict["x"], dict["y"], dict["z"], dict["free"])
    manual_method.manual_method(dict["x"], dict["y"], dict["z"], dict["free"])

    return dict["x"], dict["y"], dict["z"], dict["free"]


main("equations")
