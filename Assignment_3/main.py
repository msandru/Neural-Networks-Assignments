import numpy as np
import pickle, gzip, numpy as np

from Assignment_3 import functions as functions

apha = 0.05


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin')
train_set, valid_set, test_set
f.close()


def train():

    layer_0 = np.array(train_set[0])
    goal = np.array(train_set[1])

    weights_0_1 = np.random.normal(0, np.power(np.sqrt(784), (-1)), (100, 784))
    weights_1_2 = np.random.normal(0, np.power(np.sqrt(100), (-1)), (10, 100))

    biases_0_1 = np.random.normal(0, 1, 100)
    biases_1_2 = np.random.normal(0, 1, 10)

    losses = []
    for i in range(1):
        for index in range(len(layer_0)):
            layer_1 = functions.sigmoid(weights_0_1.dot(layer_0[index]) + biases_0_1)
            layer_2 = functions.softmax(weights_1_2.dot(layer_1) + biases_1_2)

            output = np.zeros(10)
            output[goal[index]] = 1

            loss = functions.cross_entropy(layer_2, output)
            loss += functions.L2_regularization(0.01, weights_0_1, weights_1_2)  # lambda
            losses.append(loss)

            dropout_mask = np.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask * 2

            layer_2_delta = layer_2 - output
            layer_2_delta = layer_2_delta.reshape((10, 1))

            layer_1_delta = layer_2_delta.reshape((1, 10)).dot(weights_1_2) * (layer_1 * (1 - layer_1))
            layer_1_delta = layer_1_delta.reshape((100, 1))
            layer_1_delta *= dropout_mask.reshape((100, 1))

            weights_1_2_gradient = layer_2_delta * layer_1
            weights_0_1_gradient = layer_1_delta * layer_0[index]

            weights_1_2 += -apha * weights_1_2_gradient
            weights_0_1 += -apha * weights_0_1_gradient

            biases_1_2 += -apha * layer_2_delta.flatten()
            biases_0_1 += -apha * layer_1_delta.flatten()

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

    print("Success Rate: {}".format((good_results * 100) / len(test_set[0])))

def main():
    weights_0_1, biases_0_1, weights_1_2, biases_1_2 = train()
    test(weights_0_1, biases_0_1, weights_1_2, biases_1_2)


main()
