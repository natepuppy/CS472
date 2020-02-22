import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import copy
from tools.arff import Arff
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib import pyplot as plt
import sklearn
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier as SK_MLPClassifier


### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.


class MLPClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True, deterministic=-1, num_output_nodes=1, epoch_range=10, epsilon_comparison_value=.001):
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.weights = []
        self.epochs_completed = 0
        self.prev_change_in_weights = None
        self.deterministic = deterministic
        self.num_output_nodes = num_output_nodes
        self.lossesTRS = []
        self.accuraciesTRS = []
        self.lossesVS = []
        self.accuraciesVS = []
        self.epoch_range = epoch_range
        self.epsilon_comparison_value = epsilon_comparison_value

    def fit(self, X, y, initial_weights=None):
        self.weights = self.initialize_weights(len(X[0]), self.num_output_nodes) if not initial_weights else initial_weights

        if self.deterministic == -1:
            X1, X2, y1, y2 = train_test_split(X, y, test_size=0.30)
            past_weights = []

            while True:
                self.epochs_completed += 1
                if self.shuffle:
                    X1, y1 = self._shuffle_data(X1, y1)
                    X2, y2 = self._shuffle_data(X2, y2)
                for i in range(len(X1)):
                    one_hot_encoding, final_layer_sigma, all_sigma_values = self.predictOneRow(X1[i])
                    self.back_propogate(all_sigma_values, y1[i])
                past_weights.append(copy.deepcopy(self.weights))

                self.lossesTRS.append(self.MSE(X1, y1))
                # self.accuraciesTRS.append(self.score(X1, y1))
                self.lossesVS.append(self.MSE(X2, y2))
                # self.accuraciesVS.append(self.score(X2, y2))

                if self.checkLosses2(self.lossesVS):
                    break
                    # self.weights = past_weights[len(past_weights) - 5]


        else:
            for epoch in range(self.deterministic):
                self.epochs_completed += 1
                if self.shuffle:
                    X, y = self._shuffle_data(X, y)
                for i in range(len(X)):
                    one_hot_encoding, final_layer_sigma, all_sigma_values = self.predictOneRow(X[i])
                    self.back_propogate(all_sigma_values, y[i])
                self.lossesTRS.append(self.MSE(X, y))
                self.accuraciesTRS.append(self.score(X, y))


    def predict(self, X):
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



    # # Check if the last few score are the same
    # def checkLosses(self, losses):
    #     epochRange = self.epoch_range  # 10
    #     # If loss isnt improving either, stop
    #     if len(losses) > epochRange:
    #         print("Losses: ", losses)
    #         last = losses[-epochRange:]
    #         index_of_min = np.argmin(last)
    #         if index_of_min != 0:
    #             return True
    #     else:
    #         print("Losses: ", losses)
    #     return False


    # def checkLosses2(self, losses):
    #     epoch_range = self.epoch_range
    #     if len(losses) > epoch_range:
    #         last = losses[-epoch_range:]
    #         print("accuracies: ", last)
    #         for i in range(len(last) - 1):
    #             if not round(last[i], 2) == round(last[i + 1], 2):
    #                 return False
    #         return True
    #     return False




    # def checkLosses2(self, losses):
    #     epoch_range = self.epoch_range
    #     if len(losses) > epoch_range:
    #         last = losses[-epoch_range:]
    #         # print("accuracies: ", last)
    #         differences = 0
    #         for i in range(len(last) - 1):
    #             differences += last[i + 1] - last[i]
    #         # avg = differences / len(last) - 1
    #         print("Differences: ", differences)
    #         if abs(differences) < .0001:
    #             return True
    #     return False


    def checkLosses2(self, losses):
        epoch_range = self.epoch_range
        if len(losses) > epoch_range:
            last = losses[-epoch_range - 1:]
            avg1 = 0
            avg2 = 0
            for i in range(len(last)):
                if i != len(last) - 1:
                    avg1 += last[i]
                if i != 0:
                    avg2 += last[i]

            avg1 = avg1 / self.epoch_range
            avg2 = avg2 / self.epoch_range
            print(avg1 - avg2)
            if abs(avg1 - avg2) < self.epsilon_comparison_value:
                return True
        return False





    # # Check if the last few score are the same
    # def checkLosses(self, losses):
    #     epochRange = self.epoch_range  # 10
    #     # If loss isnt improving either, stop
    #     if len(losses) > epochRange:
    #         print("Losses: ", losses)
    #         last = losses[-epochRange:]
    #         index_of_min = np.argmin(last)
    #         if index_of_min == 0:
    #             return True
    #         for i in range(len(last) - 1):
    #             if not round(last[i], 2) == round(last[i + 1], 2):
    #                 return False
    #         return True
    #     else:
    #         print("Losses: ", losses)
    #     return False
    #
    #
    # def checkLosses2(self, losses):
    #     epoch_range = self.epoch_range
    #     if len(losses) > epoch_range:
    #         last = losses[-epoch_range:]
    #         print("accuracies: ", last)
    #         for i in range(len(last) - 1):
    #             if not round(last[i], 2) == round(last[i + 1], 2):
    #                 return False
    #         return True
    #     return False
    #






    def get_one_hot_encoding(self, in_array):
        list_indexes = np.argmax(in_array)
        b = np.zeros(len(in_array))
        b[list_indexes] = 1
        return b

    """ Return accuracy of model on a given dataset. Must implement own score function.
    Args:
        X (array-like): A 2D numpy array with data, excluding targets
        y (array-like): A 2D numpy array with targets
    Returns:
        score : float
            Mean accuracy of self.predict(X) wrt. y.
    """
    def score(self, X, y):
        results = []
        one_hot_encodings, actual_values = self.predict(X)
        # value_to_use = actual_values

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
        value_to_use = actual_values
        for i in range(len(value_to_use)):
            for j in range(len(value_to_use[i])):
                results.append((value_to_use[i][j] - y[i][j]) ** 2)
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

    def _shuffle_data(self, X, y):
        combined_array = np.column_stack((X, y))
        np.random.shuffle(combined_array)
        A = []
        B = []
        for i in range(len(combined_array)):
            a, b = self.split_list(combined_array[i], (len(combined_array[i]) - self.num_output_nodes))
            A.append(a)
            B.append(b)
        return A, B

    def split_list(self, list, length):
        return list[:length], list[length:]

    def add_bias(self, X):
        biases = np.ones((len(X), 1))
        return np.hstack((X, biases))

    def printWeights(self):
        for j in range(len(self.weights)):    # for each layer
            print("Layer ", j + 1)
            for k in range(len(self.weights[j])):
                for w in range(len(self.weights[j][k])):
                    print(np.format_float_scientific(self.weights[j][k][w], precision=2))
            print()



def targets_to_one_hot_encoding(y, num_classes):
    return_array = []
    for i in range(len(y)):
        arr = np.zeros(num_classes)
        arr[int(round(y[i][0]))] = 1
        return_array.append(arr)
    return return_array


def plotBar(data, labels, xlabel="", ylabel="", title=""):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(len(labels)), labels)
    if type(data) == dict:
        width = 1 / (len(data) + 1)
        for i, key in enumerate(data):
            value = data[key]
            offset = width * (i - len(data) // 2)
            plt.bar(np.arange(len(value)) + offset, value, width=width, label=key)
        plt.legend()
    else:
        plt.bar(range(len(data)), data)
    plt.show()


def plotLine(data, data2, xlabel="", ylabel="", title=""):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(data, label='TS MSE')
    plt.plot(data2, label="VS MSE")
    plt.legend()
    plt.show()


# Graph number 2
# learning_rates = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 5]
# epochs = [122, 116, 88, 38, 26, 41, 16]
# plotBar(epochs, learning_rates, xlabel="Learning Rate", ylabel="Epochs", title="Vowel Learning Rate vs. Epochs")





# This is for graph number 3
# training_loss = [0.06, 0.03, 0.03, 0.04, 0.05, 0.06, 0.09]
# test_loss = [0.06, 0.04, 0.04, 0.05, 0.06, 0.06, 0.09]
# validation_loss = [0.06, 0.04, 0.04, 0.05, 0.05, 0.06, 0.09]
#
# ind = np.arange(7)
# training_loss = (0.06, 0.03, 0.03, 0.04, 0.05, 0.06, 0.09)
# test_loss = (0.06, 0.04, 0.04, 0.05, 0.06, 0.06, 0.09)
# validation_loss = (0.06, 0.04, 0.04, 0.05, 0.05, 0.06, 0.09)
#
# plt.bar(ind, training_loss, 0.15, color='#ff0000',label='MSE TRS')
# plt.bar(ind + 0.15, test_loss, 0.15, color='#00ff00', label='MSE TS')
# plt.bar(ind + 0.30, validation_loss, 0.15, color='#0000ff', label='MSE VS')
#
# plt.xlabel('Learning Rate')
# plt.ylabel('MSE')
# plt.xticks(ind+0.15, ("0.01", "0.05", "0.1", "0.5", "1", "1.5", "5"))
# plt.title("Vowel MSE vs. Learning Rate")
# plt.legend()
# plt.show()
# print()




# This is for graph number 5
# ind = np.arange(7)
# training_loss =   (0.081, 0.074, 0.065, 0.045, 0.037, 0.025, .023)
# test_loss =       (0.084, 0.078, 0.071, 0.052, 0.045, 0.031, .030)
# validation_loss = (0.082, 0.075, 0.071, 0.053, 0.042, 0.032, .030)
#
# plt.bar(ind, training_loss, 0.15, color='#ff0000',label='MSE TRS')
# plt.bar(ind + 0.15, test_loss, 0.15, color='#00ff00', label='MSE TS')
# plt.bar(ind + 0.30, validation_loss, 0.15, color='#0000ff', label='MSE VS')
#
# plt.xlabel('Hidden Layer Nodes')
# plt.ylabel('MSE')
# plt.xticks(ind+0.15, ("1", "2", "4", "8", "16", "32", "64"))
# plt.title("Vowel MSE vs. # Nodes in Hidden Layer")
# plt.legend()
# plt.show()
# print()


# This is for graph number 6
ind = np.arange(7)

num_epochs = (12, 17, 19, 23, 14, 20, 11)
plt.bar(ind, num_epochs, 0.15, color='#ff0000')

plt.xlabel('Momentum')
plt.ylabel('Epochs')
plt.xticks(ind+0.15, ("0.0", "0.1", "0.2", "0.4", "0.6", "0.8", "1.0"))
plt.title("Vowel Epochs vs. Momentum")
plt.legend()
plt.show()
print()




# Files to be read
arff_files = [
    # "iris",
    "vowel",
    # "linsep2nonorigin",
    # "data_banknote_authentication",
]

# Hyper-parameters
learning_rate = 0.1
momentum = 0.8
deterministic = -1
shuffle = False
split_data = True
test_percentage = 0.25
hidden_layer_widths = [32]
epoch_range = 10
use_scikit_learn = False
epsilon_comparison_value = 0.001


if use_scikit_learn:
    # This section runs the voting data on the scikit learn perceptron
    fileName = "vowel.arff"
    mat = Arff(fileName,label_count=1)
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1,1)
    hidden_layer_widths.append((2 * len(data[0])) + 1)

    clf = SK_MLPClassifier(hidden_layer_sizes=hidden_layer_widths, activation='relu', solver='adam', alpha=0.0001,batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
    X1, X2, y1, y2 = train_test_split(data, labels, test_size=test_percentage)
    result = clf.fit(X1, y1)
    prd_r = clf.predict(X2)
    # test_acc = accuracy_score(ts_y, prd_r) * 100.
    loss_values = clf.loss_curve_
    print(loss_values)


for i in range(len(arff_files)):
    # Get the file
    fileName = arff_files[i] + ".arff"
    mat = Arff(fileName, label_count=1)

    # Parse and prep the data the data / instantiate network
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)

    if fileName == "vowel.arff":
        data = np.delete(data, 1, 1)  # delete second column of C
        data = np.delete(data, 0, 1)  # delete second column of C
        data = np.delete(data, 0, 1)  # delete second column of C

    # hidden_layer_widths.append((2 * len(data[0])) + 1)

    num_output_nodes = len(mat.enum_to_str[len(mat.enum_to_str) - 1])
    labels = targets_to_one_hot_encoding(labels, num_output_nodes)
    PClass = MLPClassifier(hidden_layer_widths, learning_rate, momentum, shuffle, deterministic, num_output_nodes, epoch_range, epsilon_comparison_value)
    data = PClass.add_bias(data)
    print(fileName)

    # Run the data either split or not split
    if not split_data:
        PClass.fit(data, labels)
        accuracy = PClass.score(data, labels)
        loss = PClass.MSE(data, labels)
        print("Epochs Completed: ", PClass.epochs_completed)
        print("Accuray = {:.2f}".format(accuracy))
        print("Loss = {:.2f}".format(loss))
        print("\n\nFinal Weights: ")
        PClass.printWeights()
        print("Done")

    else:
        # FIXME make sure shuffle and split data work  (Maybe refactor this to make it more concise)
        X1, X2, y1, y2 = train_test_split(data, labels, test_size = test_percentage)
        PClass.fit(X1, y1)
        train_accuracy = PClass.score(X1, y1)
        test_accuracy = PClass.score(X2, y2)
        train_loss = PClass.MSE(X1, y1)
        test_loss = PClass.MSE(X2, y2)
        # plotLine(PClass.lossesTRS, PClass.lossesVS, xlabel="Epoch", ylabel="MSE", title="Iris MSE Across Epochs")
        # plotLine(PClass.accuraciesTRS, PClass.accuraciesVS, xlabel="Epoch", ylabel="Accuracy", title="Iris Accuracy Across Epochs")
        print("Epochs -- momentum")
        print("Epochs: ", PClass.epochs_completed)
        print("Momentum: ", momentum)

        # print("Epochs Completed: ", PClass.epochs_completed)
        # print("Training Accuray = {:.2f}".format(train_accuracy))
        # print("Test Accuray = {:.2f}".format(test_accuracy))
        # print("Training Loss = {:.2f}".format(train_loss))
        # print("Test Loss = {:.2f}".format(test_loss))
        # print("Validation Accuray = {:.2f}".format(PClass.lossesVS[len(PClass.accuraciesVS) - 1]))
        # print("Validation Loss = {:.2f}".format(PClass.lossesVS[len(PClass.lossesVS) - 1]))
        # print("Learning Rate: ", learning_rate)
        # print("\n\nFinal Weights: ")
        # PClass.printWeights()

print("Done")











# # This is for graph number 6
# ind = np.arange(7)

# num_epochs = (12, 17, 19, 23, 14, 20, 11)
# plt.bar(ind, training_loss, 0.15, color='#ff0000',label='MSE TRS')

# plt.xlabel('Momentum')
# plt.ylabel('Epochs')
# plt.xticks(ind+0.15, ("0.0", "0.1", "0.2", "0.4", "0.6", "0.8", "1.0"))
# plt.title("Vowel Epochs vs. Momentum")
# plt.legend()
# plt.show()
# print()





# Epochs:  12
# Momentum:  0.0

# Epochs:  17
# Momentum:  0.1

# Epochs:  19
# Momentum:  0.2

# Epochs:  23
# Momentum:  0.4

# Epochs:  14
# Momentum:  0.6

# Epochs:  20
# Momentum:  0.8

# Epochs:  11
# Momentum:  1.0