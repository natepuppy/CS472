import numpy as np
from tools.arff import Arff
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

class PerceptronClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, lr=.1, shuffle=True, deterministic=-1):
        """ Initialize class with chosen hyperparameters.
        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.weights = []
        self.epochs_completed = 0


    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.weights = self.initialize_weights(len(X[0])) if not initial_weights else initial_weights

        # Executes for non-deterministic
        if self.deterministic == -1:
            scores = []
            while True:
                if self.shuffle:
                    self._shuffle_data(X, y)
                for i in range(len(X)):
                    output = self.predict(X[i])
                    self.update_weights(y[i][0], output, X[i])
                self.epochs_completed += 1
                score = self.score(X, y)
                scores.append(score)
                if self.checkScores(scores):
                    break
        # Executes for deterministic
        else:
            for j in range(self.deterministic):
                if self.shuffle:
                    self._shuffle_data(X, y)
                for i in range(len(X)):
                    output = self.predict(X[i])
                    self.update_weights(y[i][0], output, X[i])
                self.epochs_completed += 1
        return self

    # Check if the last five score are the same
    def checkScores(self, scores):
        if len(scores) > 5:
            last5 = scores[-5:]
            for i in range(len(last5) - 1):
                if not round(last5[i], 2) == round(last5[i + 1], 2):
                    return False
            return True
        return False

    def update_weights(self, T, Z, X):
        scalar = self.lr * (T - Z)
        change = [x * scalar for x in X]
        self.weights = np.add(self.weights, change)



    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        sum = 0
        for i in range(len(X)):
            sum = sum + (X[i] * self.weights[i])
        if sum > 0:
            output = 1.0
        else:
            output = 0.0
        return output



    def initialize_weights(self, length):
        """ Initialize weights for perceptron. Don't forget the bias!  """
        for i in range(length):
            self.weights.append(0)
        return self.weights


    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        results = []
        outputs = []

        for i in range(len(X)):
            output = self.predict(X[i])
            outputs.append(output)

            if output == y[i][0]:
                results.append(1.0)
            else:
                results.append(0.0)

        return sum(results) / len(results)

    def add_bias(self, X):
        biases = np.ones((len(X), 1))
        return np.hstack((X, biases))



    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
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

    # Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights

    # Splits the data into a training and a test set
    def split_data(self, percent_training, X, y):
        A, B = self._shuffle_data(X, y)
        length_A = float(len(A))
        numItems = int(length_A * percent_training)
        X1, X2 = self.split_list(A, numItems)
        y1, y2 = self.split_list(B, numItems)
        return X1, y1, X2, y2

    # Prints the weights in a better format than the default boiler-plate
    def printWeights(self,):
        print("[", end='')
        for i in range(len(self.weights)):
            if i == (len(self.weights) - 1):
                print("{:5.3f}]".format(self.weights[i]))
                print()
            else:
                print("{:5.3f},".format(self.weights[i]), "", end = '')
            if i % 10 == 0 and i != 0:
                print()

# Files to be read
arff_files = [
    "seperable",
    "StandardVoting",
    "non_seperable",
    "linsep2nonorigin",
    "data_banknote_authentication",
]

# Hyper-parameters
learning_rate = .1
shuffle = True
deterministic = 10   # -1 indicates to run nondeterministically

split_data = True
training_percentage = .7

sckikitLearn = True

for i in range(len(arff_files)):
    # Get the file
    fileName = arff_files[i] + ".arff"
    mat = Arff(fileName,label_count=1)

    # Parse the data
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1,1)
    PClass = PerceptronClassifier(learning_rate, shuffle, deterministic)
    data = PClass.add_bias(data)
    print(fileName)

    # Run the data either split or not split
    if not split_data:
        PClass.fit(data,labels)
        accuracy = PClass.score(data,labels)
        print("Epochs Completed: ", PClass.epochs_completed)
        print("Accuray = [{:.2f}]".format(accuracy))
        print("Final Weights = ", end='')
        PClass.printWeights()
    else:
        # X1, y1, X2, y2 = PClass.split_data(training_percentage, data, labels)
        X1, X2, y1, y2 = train_test_split(data, labels, test_size = 0.33)
        PClass.fit(X1, y1)
        train_accuracy = PClass.score(X1, y1)
        test_accuracy = PClass.score(X2, y2)
        print("Epochs Completed: ", PClass.epochs_completed)
        print("Training Accuray = [{:.2f}]".format(train_accuracy))
        print("Test Accuray = [{:.2f}]".format(test_accuracy))
        print("Final Weights = ", end='')
        PClass.printWeights()





# This section runs the voting data on the scikit learn perceptron
# fileName = "StandardVoting.arff"
# mat = Arff(fileName,label_count=1)
# data = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
#
# scikit_perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=1000,
#             tol=0.001, shuffle=True, verbose=0, eta0=1.0, n_jobs=None,
#             random_state=0, early_stopping=False, validation_fraction=0.1,
#             n_iter_no_change=5, class_weight=None, warm_start=False)
#
#
# result = scikit_perceptron.fit(data, labels)
# print(result)
