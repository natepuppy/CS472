import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tools.arff import Arff
import math as math

class KNNClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,columntype=[],weight_type='inverse_distance',k_value=3): ## add parameters here
        self.columntype = columntype #Note This won't be needed until part 5
        self.weight_type = weight_type
        self.k_value = k_value

    def fit(self,X,y):
        self.X = X
        self.y = y
        return self


    def predict(self, data):
        return_values = []

        for i in range(len(data)):  # each instance in the test set
            distances = []
            indexes = []
            for j in range(len(self.X)):   # each instance in the train set
                dist = self.calculate_distance(data[i], self.X[j])

                if len(distances) < self.k_value:
                    distances.append(dist)
                    indexes.append(j)
                else:
                    max_value_index = int(np.argmax(distances))
                    if distances[max_value_index] > dist:
                        distances[max_value_index] = dist
                        indexes[max_value_index] = j

            # now that we have the k closest instances, find the correct output class, or regression value

            if self.columntype[-1] != 'nominal':
                output = self.determine_regression_prediction(distances, indexes)
            else:
                output = self.determine_nominal_class(distances, indexes)

            return_values.append(output)
        return return_values

    # out_classes = []
    # for k in range(len(indexes)):
    #     index = indexes[k]
    #     out_class = self.y[index]
    #     out_classes.append(out_class)
    # values, counts = np.unique(out_classes, return_counts=True)
    # ind = np.argmax(counts)
    # return_values.append(values[ind])

    def calculate_distance(self, arr1, arr2):
        total = 0
        for k in range(len(arr1)):
            dist = 0
            if self.columntype == 'nominal':   # find distance for two nominal values
                if arr1[k] != arr2[k]:
                    dist = 1
            else:
                dist = (arr1[k] - arr2[k]) ** 2
            total += dist

        total += np.sqrt(total)
        if self.weight_type == 'inverse_distance':
            if total == 0:
                return 0
            return 1 / (total ** 2)
        return total

    def determine_nominal_class(self, distances, indexes):
        if self.weight_type == 'inverse_distance':

        out_classes = []
        for k in range(len(indexes)):
            index = indexes[k]
            out_class = self.y[index]
            out_classes.append(out_class)
        values, counts = np.unique(out_classes, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]

    def determine_regression_prediction(self, distances, indexes):
        numerator = 0
        denominator = 0
        for k in range(len(indexes)):
            index = indexes[k]
            numerator += self.y[index] * distances[k]
            denominator += distances[k]
        return numerator / denominator


            # if self.weight_type == 'inverse_distance':
            #     # dist = 1 / ((np.sqrt(np.sum((data[i] - self.X[j]) ** 2))) ** 2)
            # else:
            #     # dist = np.sqrt(np.sum((data[i] - self.X[j]) ** 2))





    # def euclidean_distance_2(self, x1, y1, x2, y2):
    #     math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))





    def score(self, X, y):
        predictions = self.predict(X)
        total = 0
        for i in range(len(y)):
            print(y[i][0], " - ", predictions[i])
            if abs(y[i][0] - predictions[i]) < .001:
                total += 1
            else:
                total += 0
        return total / len(y)



def normalize(X, column_types):
    num_attributes = len(X[0])
    num_instances = len(X)
    col_max = X.max(axis=0)
    col_min = X.min(axis=0)

    for i in range(num_attributes):
        if column_types[i] == 'nominal':
            continue
        for j in range(num_instances):
            X[j][i] = (X[j][i] - col_min[i]) / (col_max[i] - col_min[i])
    return X


def start():
    # Files to be read
    arff_files = [
        "credit_approval",      # 0
        "diabetes",             # 1
        "diabetes_test",        # 2
        "housing_test",         # 3
        "housing_train",        # 4
        "seismic-bumps_test",   # 5
        "seismic-bumps_train",  # 6
        "telescope_test",       # 7
        "telescope_train",      # 8
    ]

    # hyper-parameters
    test_index = 5
    train_index = 6
    weight_type = 'distance'   # inverse_distance or distance
    k_value = 3
    regression = False
    normalization = False

    # Get the file and Parse the data
    file_test = arff_files[test_index] + ".arff"
    file_train = arff_files[train_index] + ".arff"
    print("file_test: ", file_test, "file_train: ", file_train)
    mat_test = Arff(file_test, label_count=1)
    mat_train = Arff(file_train, label_count=1)
    data_test = mat_test.data[:,0:-1]
    labels_test = mat_test.data[:,-1].reshape(-1,1)
    data_train = mat_train.data[:,0:-1]
    labels_train = mat_train.data[:,-1].reshape(-1,1)
    column_types = mat_test.attr_types

    if normalization:
        data_test = normalize(data_test, column_types)
        data_train = normalize(data_train, column_types)

    # Train and Score
    PClass = KNNClassifier(columntype=column_types, weight_type=weight_type, k_value=k_value)
    PClass.fit(data_train, labels_train)
    score = PClass.score(data_test, labels_test)
    print(score)

start()


# TODO
# How to handle real outputs
# is normalization only for the input features? -- yes i think
# normalize test and train?




















"""
git add .
git commit -m "Update for KNN"
git push
"""





"""
Args:
    columntype for each column tells you if continues[real] or if nominal[categoritcal].
    weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
"""