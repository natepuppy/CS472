import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tools.arff import Arff
import math as math


# TODO
# How to handle nominal attributes
# How to handle real outputs
# k_value == ???    (Parameter?)
# normalization option (x-xmin)/(xmax-xmin)   (Pre-Process???)
# is fit right??




class KNNClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,columntype=[],weight_type='inverse_distance'): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal[categoritcal].
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype #Note This won't be needed until part 5
        self.weight_type = weight_type



    def fit(self,X,y):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.X = X      # TODO: what else besides this?? IS this the best way??
        self.y = y
        return self


    def predict(self, data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        return_values = []

        for i in range(len(data)):
            distances = []
            indexes = []
            for j in range(len(self.X)):
                if self.weight_type == 'inverse_distance':
                    dist = 1 / ((np.sqrt(np.sum((data[i] - self.X[j]) ** 2))) ** 2)
                else:
                    dist = np.sqrt(np.sum((data[i] - self.X[j]) ** 2))

                if len(distances) < 3:
                    distances.append(dist)
                    indexes.append(j)
                else:
                    max_value_index = int(np.argmax(distances))
                    if distances[max_value_index] > dist:
                        distances[max_value_index] = dist
                        indexes[max_value_index] = j
            out_classes = []
            for k in range(len(indexes)):
                index = indexes[k]
                out_class = self.y[index]
                out_classes.append(out_class)
            values, counts = np.unique(out_classes, return_counts=True)
            ind = np.argmax(counts)
            return_values.append(values[ind])










            # np.sqrt(numpy.sum((A - B) ** 2))
        pass

        # Returns the Mean score given input data and labels
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
        Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
        """

        return 0

    # def euclidean_distance_2(self, x1, y1, x2, y2):
    #     math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    #
    # def manhattan_distance_2(self, x1, y1, x2, y2):
    #     math.abs(((x2 - x1) ** 2)) + math.abs(((y2 - y1) ** 2))


def start():
    # Files to be read
    arff_files = [
        "credit_approval",
        "diabetes",
        "diabetes_test",
        "housing_test",
        "housing_train",
        "seismic-bumps_test",
        "seismic-bumps_train",
        "telescope_test",
        "telescope_train",
    ]

    for i in range(len(arff_files)):
        # Get the file and Parse the data
        fileName = arff_files[i] + ".arff"
        print(fileName)
        mat = Arff(fileName,label_count=1)
        data = mat.data[:,0:-1]
        labels = mat.data[:,-1].reshape(-1,1)

        PClass = KNNClassifier()
        PClass.fit(data, labels)
        score = PClass.score(data, labels)

start()
