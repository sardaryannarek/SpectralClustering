# SpectralClustering
Implementation of SpectralClustering

SpectralClustering(n_cluster, affinity='rbf', n_neighbors=10, gamma=1, assign_labels='kmeans').

n_clusters int  ։  The dimension of the projection subspace.

affinity str, default=’rbf’։ How to construct the affinity matrix.
                             ‘nearest_neighbors’: construct the affinity matrix by computing a graph of nearest neighbors.
                             ‘rbf’: construct the affinity matrix using a radial basis function (RBF) kernel.

n_neighbors int, default=10  ։ Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method. Ignored for affinity='rbf'.
assign_labels{‘kmeans’, ‘cluster_qr’}, default=’kmeans’ ։The strategy for assigning labels in the embedding space. 
