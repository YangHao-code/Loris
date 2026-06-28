import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from label_model import get_wrench_label_model


class WeShapAnalysis:
    def __init__(self, train_dataset, valid_dataset, n_neighbors=5, weights="uniform", metric="euclidean"):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        n_class = train_dataset.n_class
        self.L_train = np.array(train_dataset.weak_labels)
        n, m = self.L_train.shape
        # train a KNN model to find the neighbors of each instance in the validation set
        KNN = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        label_model = get_wrench_label_model("MV")
        label_model.fit(self.L_train)
        y_train_pred = label_model.predict(self.L_train)
        KNN.fit(X=train_dataset.features, y=y_train_pred)
        self.neighbor_dists, self.neighbor_indices = KNN.kneighbors(X=valid_dataset.features, n_neighbors=n_neighbors)

        # calculate the LF vote contribution matrix using dynamic programming
        sv_pos = np.zeros((m+1, m+1))
        sv_neg = np.zeros((m+1, m+1))
        for p in range(m+1):
            for w in range(m+1):
                if p == 0:
                    sv_pos[p, w] = 0
                elif w == 0:
                    sv_pos[p, w] = (n_class - 1) / (n_class * p)
                else:
                    sv_pos[p, w] = (p/(p+w) - (p-1)/(p+w-1))/(p+w) + sv_pos[p-1,w]*(p-1)/(p+w) + sv_pos[p,w-1]*w/(p+w)

                if w == 0:
                    sv_neg[p, w] = 0
                else:
                    sv_neg[p, w] = ((p / (p + w)) - 1 / n_class - sv_pos[p, w] * p) / w

        # calculate the phi matrix of each LF
        vote_count = np.zeros((n, n_class), dtype=int)
        for c in range(n_class):
            vote_count[:, c] = np.sum(self.L_train == c, axis=1)

        total_vote_count = np.sum(vote_count, axis=1)
        phi = np.zeros((n, m, n_class), dtype=float)
        for i in range(n):
            for j in range(m):
                if self.L_train[i, j] == -1:
                    continue
                for c in range(n_class):
                    if self.L_train[i, j] == c:
                        phi[i, j, c] = sv_pos[vote_count[i, c], total_vote_count[i] - vote_count[i, c]]
                    else:
                        phi[i, j, c] = sv_neg[vote_count[i, c], total_vote_count[i] - vote_count[i, c]]

        self.phi = phi
        self.epsilon = 1e-6

    def calculate_contribution(self, indices=None):
        if indices is None:
            indices = np.arange(len(self.valid_dataset))
        n, m, n_class = self.phi.shape

        n_neighbors = self.neighbor_indices.shape[1]
        contribution = np.zeros((n, m), dtype=float)  # contribution matrix
        for idx in indices:
            y = self.valid_dataset.labels[idx]
            if self.weights == "distance":
                normalizer = np.sum(1 / (self.neighbor_dists[idx] + self.epsilon))
            elif self.weights == "uniform":
                normalizer = n_neighbors
            else:
                raise ValueError("weights should be either 'distance' or 'uniform'")

            for i, dist in zip(self.neighbor_indices[idx], self.neighbor_dists[idx]):
                if self.weights == "distance":
                    contribution[i, :] += self.phi[i, :, y] / (dist + self.epsilon) / normalizer
                elif self.weights == "uniform":
                    contribution[i, :] += self.phi[i, :, y] / normalizer

        contribution /= len(indices)
        return contribution

    def calculate_weshap_score(self, indices=None):
        contribution = self.calculate_contribution(indices)
        return np.sum(contribution, axis=0)


