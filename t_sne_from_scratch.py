import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

class TSNE:
    def __init__(self, learning_rate, n_components=2, perplexity=30, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.embedded = None

    def fit(self, data):
        n = data.shape[0]

        # Calculating squared L2 Distance matrix, dist_ij = norm(x_i,x_j)^2
        dist = pairwise_distances(data, metric='l2')
        dist *= dist
        # binary search for sigmas in range(10^-3,10^3) with eps = 10^-3 and max_iters = 300

        max_iters = 300
        eps = 10**(-3)
        sigmas = np.zeros(n)
        # binary search
        for num_sigma in range(n):
            lower = 10 ** (-3)
            upper = 10 ** 3
            for _ in range(max_iters):
                sigma = (lower + upper) * .5
                perpl = 2 ** TSNE.shannon(TSNE.cond_vector(dist[num_sigma, :], sigma))

                if perpl > self.perplexity:
                    upper = sigma
                else:
                    lower = sigma

                if upper - lower <= eps:
                    sigmas[num_sigma] = sigma
                    break
            sigmas[num_sigma] = sigma
        # calculating conditional affinities
        cond_affinities = np.zeros((n, n))
        for i, sigma in enumerate(sigmas):
            cond_affinities[i, :] = TSNE.cond_vector(dist[i, :], sigmas[i])
        np.fill_diagonal(cond_affinities, 0.)
        cond_affinities += 10**(-7)
        # calculating symmetric affinities
        symm_aff = (cond_affinities + cond_affinities.T) / (2 * n)
        y = np.random.multivariate_normal(np.zeros(self.n_components), np.identity(self.n_components,) * 10**(-4), n)
        Y = []
        Y.append(y)
        Y.append(y)

        # gradient decent
        momentum = 0.5
        for i in tqdm(range(self.n_iter)):

            dists = pairwise_distances(Y[-1], metric='l2')
            nom = 1 / (1 + dists * dists)
            np.fill_diagonal(nom, 0.)
            Q = nom / np.sum(np.sum(nom))
            # momentum update can be enhanced
            momentum = 0.5 if i < 250 else 0.8
            Y.append(Y[-1] - self.learning_rate * TSNE.gradient(symm_aff, Q, Y[-1]) + momentum * (Y[-1] - Y[-2]))
            # https://arxiv.org/pdf/1301.3342.pdf using this paper We can speed up our algorithm using Barnes-Hut approximation
            # No implementation this time :(
        self.embedded = Y[-1]

    def fit_transform(self, data):

        self.fit(data)
        return self.embedded

    @staticmethod
    def cond_vector(dist, sigma):
        # conditional probability vec for some data point x_i and sigma_i
        n = dist.shape[0]
        P = np.zeros(n)

        p = np.exp(-dist / (2 * sigma * sigma))
        normalize = np.sum(p) - 1

        return p / normalize

    @staticmethod
    def shannon(vec):
        # this gives as Shannon entropy for some prob vec
        return -1 * np.sum(vec * np.log2(vec))

    @staticmethod
    def gradient(P, Q, Y):
        # evaluating gradients for Kl
        pq = P - Q
        y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

        dist = pairwise_distances(Y, metric='l2')
        aux = 1 / (1 + dist*dist)
        return 4 * (np.expand_dims(pq, 2) * y_diff * np.expand_dims(aux, 2)).sum(1)
