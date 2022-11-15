import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import sys
path = 'output1.txt'
sys.stdout = open(path, 'w')

ATTRIBUTES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
LABELS = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
TARGET = 'class'

df = pd.read_csv('iris.data', sep=',', names=ATTRIBUTES + [TARGET])

x = df[ATTRIBUTES]
Y = df[TARGET].tolist()
y_labeled = np.array([LABELS[i] for i in Y])


def find_distance(point, set_of_points):
    print
    return np.sqrt(np.sum((point - set_of_points)**2, axis=1))


def normalized_mutual_info(currlabels: np.ndarray, actuallabels: np.ndarray, k: int):
    '''Calculate NMI using clusters and actual labels'''
    sample_size = len(actuallabels)
    cls0 = len(actuallabels[actuallabels == 0])
    cls1 = len(actuallabels[actuallabels == 1])
    cls2 = len(actuallabels[actuallabels == 2])

    h_y = 0
    if cls0:
        h_y += -(cls0/sample_size)*np.log2(cls0/sample_size)
    if cls1:
        h_y += -(cls1/sample_size)*np.log2(cls1/sample_size)
    if cls2:
        h_y += -(cls2/sample_size)*np.log2(cls2/sample_size)

    h_c = 0

    for i in range(k):
        clsi = len(currlabels[currlabels == i])
        if clsi:
            h_c += -(clsi/sample_size)*np.log2(clsi/sample_size)

    h_y_c = 0

    for i in range(k):
        idx = np.where(currlabels == i)[0]
        new_actual = actuallabels[idx]
        size = len(new_actual)
        cls0 = len(new_actual[new_actual == 0])
        cls1 = len(new_actual[new_actual == 1])
        cls2 = len(new_actual[new_actual == 2])

        temp = 0
        if cls0:
            temp += -(cls0/size)*np.log2(cls0/size)
        if cls1:
            temp += -(cls1/size)*np.log2(cls1/size)

        if cls2:
            temp += -(cls2/size)*np.log2(cls2/size)

        h_y_c += (temp*(len(idx)/len(actuallabels)))

    nmi = (2*(h_y-h_y_c))/(h_c+h_y)

    return nmi


class KMeans:
    def __init__(self, X: pd.DataFrame, k: int = 2, max_iteration: int = 1000):
        '''
        Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        Then the rest are initialized w/ probabilities proportional to their distances to the first
        Pick a random point from train data for first centroid
        '''
        self.centroids = [random.choice(X)]
        for _ in range(k-1):
            # Calculate distances from points to the centroids
            distance_array = np.sum([find_distance(centroid, X)
                            for centroid in self.centroids], axis=0)
            # Normalize the distances
            distance_array /= np.sum(distance_array)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(
                range(len(X)), size=1, p=distance_array)
            self.centroids += [X[new_centroid_idx]]
        # Iterate, adjusting centroids until converged or until passed max_iteration
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < max_iteration:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(k)]
            for x in X:
                distance_array = find_distance(x, self.centroids)
                centroid_idx = np.argmin(distance_array)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0)
                              for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                # Catch any np.nans, resulting from a centroid having no points
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        ''' 
            Return Assigned label for each data to nearest centroid index. 
        '''
        centroids = []
        centroid_idxs = []
        for x in X:
            distance_array = find_distance(x, self.centroids)
            centroid_idx = np.argmin(distance_array)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return np.array(centroids), np.array(centroid_idxs)


def plot_pca():
    pca = PCA().fit(x)

    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 5, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    print("Number of components")
    print(xi)
    print("Cumulative variance (%)")
    print(y)

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 5, step=1))
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.savefig('pca_plot.jpg')
    plt.clf()


def plot_k_vs_nmi():
    pca = PCA(0.95)
    x_pc = pca.fit_transform(x)
    x_axis = []
    y_axis = []
    for k in range(2, 9):
        kmeans = KMeans(x_pc, k=k, max_iteration=1000)
        centroids, labelings = kmeans.evaluate(x_pc)
        x_axis.append(k)
        y_axis.append(normalized_mutual_info(labelings, y_labeled, k))

    print("\nValues of K")
    print(x_axis)
    print("Normalized Mutual Info")
    print(y_axis)

    plt.plot(x_axis, y_axis)
    plt.xlabel('K')
    plt.ylabel('Normalized Mutual Information')
    plt.title('K vs Normalized Mutual Information')
    plt.savefig('k_vs_nmi.png')
    plt.clf()


def plot_data():

    # plotting data after applying PCA(0.95)
    pca = PCA(0.95)
    x_pc = pca.fit_transform(x)
    x_axis = [i[0] for i in x_pc]
    y_axis = [i[1] for i in x_pc]
    plt.scatter(x_axis, y_axis)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data Representation after applying PCA(0.95)')
    plt.savefig('PCA_data_representation.png')
    plt.clf()

    # plotting pca data after applying kmeans clustering with k = 3
    kmeans = KMeans(x_pc, k=3)
    centroids, labelings = kmeans.evaluate(x_pc)
    # filter rows of original data
    filtered_label0 = x_pc[labelings == 0]
    filtered_label1 = x_pc[labelings == 1]
    filtered_label2 = x_pc[labelings == 2]

    # Plotting the results
    plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1], color='red')
    plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1], color='black')
    plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1], color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K=3 K-Means Clustering Data Representation')
    plt.savefig('K3_cluster_data_representation.png')
    plt.clf()


if __name__ == '__main__':
    plot_pca()
    plot_data()
    plot_k_vs_nmi()
