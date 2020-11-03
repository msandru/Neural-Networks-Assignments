import pickle, gzip, numpy as np

alpha = 0.1

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin')
train_set, valid_set, test_set
f.close()

def activation(input):
    if input > 0:
        input = 1
    else:
        input = 0

    return input


def target_check(t, digit):
    if t == digit:
        return 1
    else:
        return 0


def train(weights, biases):
    nr_iterations = 10

    while (nr_iterations > 0):

        for j in range(0, 10):
            for iterator in range(0, len(train_set[0])):
                z = np.dot(train_set[0][iterator], weights[j]) + biases[j]

                output = activation(z)

                error = target_check(train_set[1][iterator], j) - output
                derivative = error * train_set[0][iterator]

                weights[j] = np.add(weights[j], derivative * alpha)
                biases[j] = biases[j] + error * alpha

        nr_iterations -= 1

def test(weights, biases):
    goodResults = 0

    for iterator in range(len(test_set[0])):
        output = []
        for j in range(0, 10):
            output.append(np.dot(test_set[0][iterator], weights[j]) + biases[j])

        if (output.index(max(output)) == test_set[1][iterator]):
            goodResults += 1

    print("good results: ", goodResults)
    print("probability: ", (goodResults * 100) / len(test_set[0]))

def main():
    weights = np.zeros((10, 784))

    biases = np.zeros(784)
    train(weights, biases)
    test(weights, biases)

main()
