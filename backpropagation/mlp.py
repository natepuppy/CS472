import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import copy
from tools.arff import Arff
from sklearn.model_selection import train_test_split

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

# You can get away with doing just the one layer
# The output of a bias node is always one, then it is multiplied by the weight
# There is never an input to a bias node
# input_layer_number does not include bias node



class MLPClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True, deterministic=-1, num_output_nodes=1):
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.weights = []
        self.epochs_completed = 0
        self.prev_change_in_weights = None
        self.deterministic = deterministic
        self.num_output_nodes = num_output_nodes

    print("\n\n\n\nNew File: ")

    def fit(self, X, y, initial_weights=None):
        self.weights = self.initialize_weights(len(X[0]), self.num_output_nodes) if not initial_weights else initial_weights

        for epoch in range(self.deterministic):
            # print("epoch: ", epoch + 1)
            print("Accuracy: ", self.score(X, y))
            # print("MSE Loss: ", self.MSE(X, y))
            if self.shuffle:
                X, y = self._shuffle_data(X, y)
            for i in range(len(X)):
                one_hot_encoding, final_layer_sigma, all_sigma_values = self.predictOneRow(X[i])
                self.back_propogate(all_sigma_values, y[i])


    def predict(self, X):
        # FIXME What is this supposed to mean??  array, shape (n_samples,)
        #  FIXME, need to implement better
        # not the one hot, just an array of all the guesses

        actual_values = []
        one_hot_encodings = []
        for i in range(len(X)):
            one_hot_encoding, final_layer_sigma, all_sigma_values = self.predictOneRow(X[i])   # This function does not use sigma_values
            one_hot_encodings.append(one_hot_encoding)
            actual_values.append(final_layer_sigma)
        return one_hot_encodings, actual_values

    def predictOneRow(self, X):
        input_for_next_layer = X  # This is the input 0, 1, 1  # y[i] is the target
        sigmas_for_full_network = []
        sigmas_for_full_network.append(copy.deepcopy(X))
        for j in range(len(self.weights)):  # Go through each layer of weights
            sigmas = []
            for k in range(len(self.weights[j])):  # Go through each row of the weights for that layer
                sigmas.append(self.sigmoid(self.multiply_vectors(input_for_next_layer, self.weights[j][k])))  # These are the sigma outputs
            if j != len(self.weights) - 1:  # append bias # don't add to the very last one (the output layer)
                sigmas.append(1.0)
            input_for_next_layer = copy.deepcopy(sigmas)
            sigmas_for_full_network.append(input_for_next_layer)

        final_sigma_values = sigmas_for_full_network[len(sigmas_for_full_network) - 1]
        one_hot_encoding = self.get_one_hot_encoding(final_sigma_values)

        return one_hot_encoding, final_sigma_values, sigmas_for_full_network


        # FIXME what if it is a tie?? Should it output [1,0,0] or [.9, ,8, .1]
        # FIXME What do I use to calculate MSE??
        # FIXME accuracy and loss???

        # # FIXME ---------------------------------------
        # if self.num_output_nodes == 1:
        #     finalValue = all_sigmas[len(all_sigmas) - 1][0]
        #     if finalValue > 0.5:
        #         output = [1.0, 0.0]
        #     else:
        #         output = [0.0, 1.0]

    def get_one_hot_encoding(self, in_array):
        list_indexes = np.argmax(in_array)
        b = np.zeros(len(in_array))
        b[list_indexes] = 1
        return b



        # a = np.array(in_array)
        # b = np.zeros((a.size, a.max() + 1.0))
        # b[np.arange(a.size), a] = 1.0
        # return b



    """ Return accuracy of model on a given dataset. Must implement own score function.
    Args:
        X (array-like): A 2D numpy array with data, excluding targets
        y (array-like): A 2D numpy array with targets
    Returns:
        score : float
            Mean accuracy of self.predict(X) wrt. y.
    """

    # FIXME --------------------------------------- Implement score   (MSE (VS accuracy))   Could do accuracy (One Hot encoding) and MSE
    # returns mean accuracy
    def score(self, X, y):
        results = []
        one_hot_encodings, actual_values = self.predict(X)

        for i in range(len(one_hot_encodings)):
            for j in range(len(one_hot_encodings[i])):
                if one_hot_encodings[i][j] != y[i][j]:
                    results.append(0.0)
                    break
                if j == len(one_hot_encodings[i]) - 1:
                    results.append(1.0)
        return sum(results) / len(results)





    def MSE(self, X, y):
        results = []
        one_hot_encodings, actual_values = self.predict(X)

        for i in range(len(actual_values)):
            for j in range(len(actual_values[i])):
                results.append((actual_values[i][j] - y[i][j]) ** 2)

        return sum(results) / len(results)







    # create a global variable with the last weight changes (d_weight)
    def back_propogate(self, sigma_values, targets):
        new_weights = copy.deepcopy(self.weights)
        previous_deltas = []

        if self.prev_change_in_weights == None:
            self.prev_change_in_weights = copy.deepcopy(self.weights)  # These values will not actually be used, this is just to make it the right size so you can loop through it
            firstIteration = True
        else:
            firstIteration = False

        for j in range(len(targets)):
            previous_deltas.append(self.calc_output_delta(targets[j], sigma_values[len(sigma_values) - 1][j]))

        for j in reversed(range(len(new_weights))):    # for each layer (backwards)
            sigmas = sigma_values[j]
            updated_deltas = []
            for k in range(len(new_weights[j])):    # For
                delta = previous_deltas[k]

                for w in range(len(new_weights[j][k])):
                    if firstIteration:
                        self.prev_change_in_weights[j][k][w] = self.calc_weight_change(self.lr, delta, sigmas[w])
                        new_weights[j][k][w] += self.calc_weight_change(self.lr, delta, sigmas[w])
                    else:
                        change = self.calc_weight_change(self.lr, delta, sigmas[w]) + (self.prev_change_in_weights[j][k][w] * self.momentum)
                        new_weights[j][k][w] += change
                        self.prev_change_in_weights[j][k][w] = change

            # get the nest delta values
            if j > 0:
                for w in range(len(self.weights[j][0]) - 1):
                    x = 0
                    for k in range(len(self.weights[j])):
                        x = x + (self.weights[j][k][w] * previous_deltas[k])
                    updated_deltas.append(self.calc_hidden_delta(x, sigmas[w]))
                previous_deltas = copy.deepcopy(updated_deltas)
        self.weights = copy.deepcopy(new_weights)

    def calc_output_delta(self, t_1, sigma_1):
        return (t_1 - sigma_1) * (sigma_1) * (1 - sigma_1)

    def calc_hidden_delta(self, prev_delta_times_weight, prev_sigma):
        return prev_delta_times_weight * (prev_sigma * (1 - prev_sigma))

    def calc_weight_change(self, learning_rate, delta, sigma):
        return learning_rate * delta * sigma

    def multiply_vectors(self, a, b):
        total = 0
        for k in range(len(a)):
            total += (a[k] * b[k])
        return total

    def sigmoid_array(self, arr):
        for index in range(len(arr)):
            arr[index] = self.sigmoid(arr[index])
        return arr

    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))

    def initialize_weights(self, length, outputLayerSize):
        # length is the input layer size
        if self.deterministic == -1:
            return self.initialize_random(length, outputLayerSize)
        else:
            return self.initialize_zeros(length, outputLayerSize)

    def initialize_zeros(self, length, outputLayerSize):
        if len(self.hidden_layer_widths) == 0:
            return np.zeros((1, 1, length))

        array = [[] for x in range(len(self.hidden_layer_widths) + 1)]

        for i in range(len(array)):
            if i == 0:
                x = self.hidden_layer_widths[i] - 1
                array[i] = np.zeros((x, length))
            else:
                if i == len(self.hidden_layer_widths):
                    array[i] = np.zeros((outputLayerSize, self.hidden_layer_widths[i - 1]))
                else:
                    array[i] = np.zeros((self.hidden_layer_widths[i] - 1, self.hidden_layer_widths[i - 1]))
        return array

    def initialize_random(self, length, outputLayerSize):
        if len(self.hidden_layer_widths) == 0:
            return np.random.randn(1, 1, length)

        array = [[] for x in range(len(self.hidden_layer_widths) + 1)]

        for i in range(len(array)):
            if i == 0:
                array[i] = np.random.randn(self.hidden_layer_widths[i] - 1, length)
            else:
                if i == len(self.hidden_layer_widths):
                    array[i] = np.random.randn(outputLayerSize, self.hidden_layer_widths[i - 1])
                else:
                    array[i] = np.random.randn(self.hidden_layer_widths[i] - 1, self.hidden_layer_widths[i - 1])
        return array

    def get_weights(self):
        return self.weights
        # if len(self.weights) == 0:
        #     return self.initialize_weights()

    def _shuffle_data(self, X, y):
        combined_array = np.column_stack((X, y))
        np.random.shuffle(combined_array)
        A = []
        B = []
        for i in range(len(combined_array)):
            a, b = self.split_list(combined_array[i], (len(combined_array[i]) - 1))
            A.append(a)
            B.append(b)
        return A, B

    def split_list(self, list, length):
        return list[:length], list[length:]

    def add_bias(self, X):
        biases = np.ones((len(X), 1))
        return np.hstack((X, biases))





def targets_to_one_hot_encoding(y, num_classes):
    return_array = []
    for i in range(len(y)):
        arr = np.zeros(num_classes)
        arr[int(round(y[i][0]))] = 1
        return_array.append(arr)
    return return_array

















# Files to be read
arff_files = [
    "seperable",
    "StandardVoting",
    "non_seperable",
    "linsep2nonorigin",
    "data_banknote_authentication",
]

# # Hyper-parameters
learning_rate = .01
momentum = 0.5
deterministic = 10
shuffle = False
split_data = False
training_percentage = 0.0
hidden_layer_widths = [5]   # Don't forget to add one for the bias
# one_hot_encoding = True







for i in range(len(arff_files)):
    # Get the file
    fileName = arff_files[i] + ".arff"
    mat = Arff(fileName, label_count=1)

    # Parse and prep the data the data / instantiate network
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    num_output_nodes = len(mat.enum_to_str[len(mat.enum_to_str) - 1])
    labels = targets_to_one_hot_encoding(labels, num_output_nodes)
    PClass = MLPClassifier(hidden_layer_widths, learning_rate, momentum, shuffle, deterministic, num_output_nodes)
    data = PClass.add_bias(data)

    print(fileName)

    # Run the data either split or not split
    if not split_data:
        PClass.fit(data, labels)
        accuracy = PClass.score(data, labels)
        loss = PClass.MSE(data, labels)
        # print("Epochs Completed: ", PClass.epochs_completed)
        # print("Accuray = [{:.2f}]".format(accuracy))
        # print("Final Weights = ", end='')
        # PClass.printWeights()
    else:
        # FIXME make sure shuffle and split data work  (Maybe refactor this to make it more concise)
        X1, X2, y1, y2 = train_test_split(data, labels, test_size = 0.33)
        PClass.fit(X1, y1)
        train_accuracy = PClass.score(X1, y1)
        test_accuracy = PClass.score(X2, y2)
        print("Epochs Completed: ", PClass.epochs_completed)
        print("Training Accuray = [{:.2f}]".format(train_accuracy))
        print("Test Accuray = [{:.2f}]".format(test_accuracy))
        print("Final Weights = ", end='')
        # PClass.printWeights()





# Data For the homework


# self.weights = self.exampleWeights()


# def exampleWeights(self):
#     # # first iteration
#     weights1 = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
#     weights2 = [[1.0, 1.0, 1.0]]
#     return [weights1, weights2]
#
#     # # second iteration
#     # weights1 = [[1, 1, 1.00113], [1, 1, 1.00113]]
#     # weights2 = [[1.00420, 1.00420, 1.00575]]
#     # return [weights1, weights2]


# # Hyper-parameters
# learning_rate = 1.0
# shuffle = False
# split_data = False
# training_percentage = .7
# sckikitLearn = False
#
#
# PClass = MLPClassifier([3], learning_rate, 0, shuffle, 1)
# PClass.fit([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], [[1.0], [0.0]])     # Both
#
# print("")
# PClass.fit([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], [1.0])   # first iteration
# PClass.fit([[0.0, 1.0, 1.0]], [0.0])   # second iteration
