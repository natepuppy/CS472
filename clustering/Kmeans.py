import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import MinMaxScaler
from tools.arff import Arff
from sklearn.cluster import KMeans

#NOTE:  option > enter to import

class KMEANSClustering(BaseEstimator,ClusterMixin):
    def __init__(self,k=3,debug=False):
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug
        self.centroids = []
        self.clusters = []
        self.cluster_indexes = []
        self.errors = []

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        # create initial list of centroids (debug or no)
        centroids = []
        if self.debug:
            for i in range(self.k):
                centroids.append(X[i])
        else:
            indexes = np.random.choice(X.shape[0], self.k, replace=False)
            for i in range(len(indexes)):
                centroids.append(X[indexes[i]])

        while True:
            clusters = [[] for x in range(self.k)]
            cluster_indexes = [[] for x in range(self.k)]

            # find distance from this centroid to all other points
            all_distances = []
            for instance in X:
                distances = []
                for centroid in centroids:
                    dist = np.linalg.norm(centroid - instance)
                    distances.append(dist)
                all_distances.append(distances)

            all_distances = np.array(all_distances)

            # find the closest centroid and add it to that cluster
            for i in range(len(all_distances)):
                closest_cluster = np.argmin(all_distances[i])
                if isinstance(closest_cluster, np.int64):
                    clusters[closest_cluster].append(X[i])
                    cluster_indexes[closest_cluster].append(i)
                else:
                    clusters[closest_cluster[0]].append(X[i])
                    cluster_indexes[closest_cluster[0]].append(i)

            # recalculate the centroid
            new_centroids = []
            for cluster in clusters:
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)

            if np.array_equal(centroids, new_centroids):
                self.centroids = centroids
                self.clusters = clusters
                self.cluster_indexes = cluster_indexes
                break
            else:
                centroids = new_centroids

        self.SSE()
        return self

    def SSE(self):
        total_errors = []
        for i in range(len(self.clusters)):
            errors = []
            for instance in self.clusters[i]:   # each instance in a cluster
                dist = np.linalg.norm(self.centroids[i] - instance)
                errors.append(dist ** 2)
            total_errors.append(sum(errors))
        self.errors = total_errors

    def save_clusters(self, filename):
        f = open(filename,"w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(sum(self.errors)))
        for i in range(len(self.centroids)):
            f.write(np.array2string(self.centroids[i],precision=4,separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.cluster_indexes[i])))
            f.write("{:.4f}\n\n".format(self.errors[i]))
        f.close()
        return self



def normalize(X):
    num_attributes = len(X[0])
    num_instances = len(X)
    col_max = np.nanmax(X, axis=0)
    col_min = np.nanmin(X, axis=0)

    for i in range(num_attributes):
        for j in range(num_instances):
            X[j][i] = (X[j][i] - col_min[i]) / (col_max[i] - col_min[i])
    return X

def start():
    # Files to be read
    arff_files = [
        "abalone",              # 0
        "seismic-bumps_train",  # 1
        "iris"                  # 2
    ]

    # Hyper-Parameters
    index = 2
    label_count = 1         # 0 includes the output, 1 excludes the output
    normalize_data = True
    k_values = [4, 4, 4, 4, 4]
    debug = False
    sk_learn = False

    # Get the file and Parse the data
    file = arff_files[index] + ".arff"
    mat = Arff(file, label_count=label_count)

    if label_count != 0:
        data = mat.data[:, 0:-label_count]
    else:
        data = mat.data

    if normalize_data:
        data = normalize(data)

    for k in k_values:
        if sk_learn:
            result = KMeans(n_clusters=k, random_state=0).fit(data)
            print()
        else:
            result = KMEANSClustering(k=k, debug=debug).fit(data).save_clusters("debug_kmeans.txt")
            # print(sum(result.errors))


def plot():
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    ind = np.arange(6)
    data = [12.143688281579719, 7.138647703985378, 6.037852001778729, 4.571923613060553, 4.439985640515282, 3.5525683030070594]
    data2 = [18.39368828157972, 7.817456892309572, 6.613823274690356, 5.103468367707373, 4.788247858095654, 3.817678797888969]

    plt.bar(ind, data, 0.30, color='#0d17d1',label='Excluding output column')
    plt.bar(ind + 0.30, data2, 0.30, color='#cf0a17', label='Including output column')

    plt.xlabel('k')
    plt.ylabel('Total SSE')
    plt.xticks(ind+0.15, ("2", "3", "4", "5", "6", "7"))
    plt.suptitle("Total K-Means SSE for different k values")
    plt.title("Iris dataset")
    plt.legend()
    plt.show()

plot()
start()









# index = 2
# label_count = 1  # 0 includes the output, 1 excludes the output
# normalize_data = True
# k_values = [2, 3, 4, 5, 6, 7]
# debug = False
# sk_learn = False
#
# 12.143688281579719
# 7.138647703985378
# 6.037852001778729
# 4.571923613060553
# 4.439985640515282
# 3.5525683030070594
# x = [12.143688281579719, 7.138647703985378, 6.037852001778729, 4.571923613060553, 4.439985640515282, 3.5525683030070594]



# index = 2
# label_count = 0         # 0 includes the output, 1 excludes the output
# normalize_data = True
# k_values = [2, 3, 4, 5, 6, 7]
# debug = False
# sk_learn = False
#
# 18.39368828157972
# 7.817456892309572
# 6.613823274690356
# 5.103468367707373
# 4.788247858095654
# 3.817678797888969
# x = [18.39368828157972, 7.817456892309572, 6.613823274690356, 5.103468367707373, 4.788247858095654, 3.817678797888969]


# index = 2
# label_count = 0         # 0 includes the output, 1 excludes the output
# normalize_data = True
# k_values = [4, 4, 4, 4, 4]
# debug = False
# sk_learn = False
#
# 6.300610869554293
# 6.855806397983419
# 6.613823274690356
# 6.620314390462653
# 6.624080953040391
# x = [6.300610869554293, 6.855806397983419, 6.613823274690356, 6.620314390462653, 6.624080953040391]

# label_count = 1
# 6.209222461782501
# 6.038907823608437
# 5.532831003081898
# 5.541657763082495
# 6.068688762623882
# x = [6.209222461782501, 6.038907823608437, 5.532831003081898, 5.541657763082495, 6.06868]8762623882]








# def save_clusters(self,filename):
#     f = open(filename,"w+")
#     for i in range(len(self.centroids)):
#         f.write("Centroid:" + str(i) + " = ")
#         f.write(np.array2string(self.centroids[i],precision=4,separator=","))
#         cluster_string = "Cluster:" + str(i) + " "
#         f.write("\n" + cluster_string + "Size = ")
#         f.write("{:d}".format(len(self.clusters[i])))
#         f.write(" SSE = ")
#         f.write("{:.3f}\n".format(self.errors[i]))  # sse of cluster
#         f.write(cluster_string + "contains: ")
#         arr = np.array(self.cluster_indexes[i])
#         my_string = ', '.join(map(str, arr))
#         f.write("[" + my_string + "]")
#         f.write("\n")
#     f.write("SSE: {:.3f}\n".format(sum(self.errors)))
#     f.close()



