import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from numpy.linalg import eig
from scipy.linalg import qr, polar

#implementation of Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)
class SpectralClustering:
    def __init__(self, n_cluster, affinity='rbf', n_neighbors=10, gamma=1, assign_labels='kmeans'):
        self.n_cluster = n_cluster
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.assign_labels = assign_labels
        self.labels_ = None

    def fit(self, data):
        if self.affinity == 'nearest_neighbors':
            graph = kneighbors_graph(data, self.n_neighbors, mode='connectivity', include_self=True)
            graph = graph.toarray()
            w = np.logical_or(graph.T, graph) * 1

        elif self.affinity == 'rbf':
            w = rbf_kernel(data, gamma=self.gamma)

        d = np.diag(np.sum(w, axis=1))
        l = d - w

        d_minus_point_five = np.zeros(d.shape)
        np.fill_diagonal(d_minus_point_five, 1 / (d.diagonal() ** 0.5))
        l_sym = d_minus_point_five @ l @ d_minus_point_five

        eig_vals, eig_vec = eig(l_sym)
        indexes = np.argsort(eig_vals)[0:self.n_cluster]
        u = eig_vec[:, indexes]

        norm = np.sum(u * u, axis=1) ** .5
        t = u / norm.reshape(-1, 1)

        if self.assign_labels == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_cluster, init="k-means++").fit(t)
            self.labels_ = kmeans.labels_

        #cluster qr from https://arxiv.org/pdf/1609.08251.pdf
        elif self.assign_labels == "cluster_qr":
            q, r, p = qr(t.T, pivoting=True)
            c = p[:self.n_cluster]
            A, B = polar(t[c, :].T)
            self.labels_ = np.argmax(np.abs(A.T @ t.T), axis=0)

    def fit_predict(self,data):
        self.fit(data)
        return self.labels_






