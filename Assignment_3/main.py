import numpy as np
import pickle, gzip, numpy as np

from Assignment_3 import functions as functions

alpha = 0.05
neurons_layer_1 = 10
regularization_parameter = 0.0000001

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin')
train_set, valid_set, test_set
f.close()


def train():
    layer_0 = np.array(train_set[0])
    goal = np.array(train_set[1])

    weights_0_1 = np.random.normal(0, np.power(np.sqrt(784), (-1)), (neurons_layer_1, 784))

    weights_1_2 = np.random.normal(0, np.power(np.sqrt(neurons_layer_1), (-1)), (10, neurons_layer_1))

    biases_0_1 = np.random.normal(0, 1, neurons_layer_1)
    biases_1_2 = np.random.normal(0, 1, 10)

    for i in range(1):

        for index in range(len(layer_0)):
            layer_1 = functions.sigmoid(weights_0_1.dot(layer_0[index]) + biases_0_1)
            layer_2 = functions.softmax(weights_1_2.dot(layer_1) + biases_1_2)

            output = np.zeros(10)
            output[goal[index]] = 1

            layer_2_delta = layer_2 - output
            layer_2_delta = layer_2_delta.reshape((1, 10))

            layer_1_delta = layer_2_delta.dot(weights_1_2) * functions.sigmoid_derivative(layer_1)
            layer_1_delta = layer_1_delta.reshape((neurons_layer_1, 1))

            weight_decay_0_1 = alpha * np.dot(layer_1_delta,
                                              np.reshape(layer_0[index], (1, 784)))

            weight_decay_1_2 = alpha * np.dot(np.reshape(layer_2_delta, (10, 1)),
                                              np.reshape(layer_1, (1, neurons_layer_1)))

            weights_0_1 *= (1 - alpha * (regularization_parameter / len(weights_0_1)))
            weights_1_2 *= (1 - alpha * (regularization_parameter / len(weights_1_2)))

            weights_0_1 -= weight_decay_0_1
            weights_1_2 -= weight_decay_1_2

            biases_0_1 -= alpha * layer_1_delta.flatten()
            biases_1_2 -= alpha * layer_2_delta.flatten()

    return weights_0_1, weights_1_2, biases_0_1, biases_1_2


def test(weights_0_1, weights_1_2, biases_0_1, biases_1_2):
    good_results = 0

    layer_0_values = np.array(test_set[0])
    target = np.array(test_set[1])
    for index in range(len(layer_0_values)):
        layer_1 = functions.sigmoid(weights_0_1.dot(layer_0_values[index]) + biases_0_1)
        layer_2 = functions.softmax(weights_1_2.dot(layer_1) + biases_1_2)
        if np.argmax(layer_2) == target[index]:
            good_results += 1

    print("Success Rate: {} %".format((good_results * 100) / len(test_set[0])))


def main():
    weights_0_1, biases_0_1, weights_1_2, biases_1_2 = train()
    test(weights_0_1, biases_0_1, weights_1_2, biases_1_2)


main()
