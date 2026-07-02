import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy


def random_argmax(arr):
    # Apply random_argmax to each row in the array
    def random_choice_of_max(arr):
        # Find the maximum value
        max_val = np.max(arr)

        # Find all indices where the array is equal to the maximum value
        max_indices = np.where(arr == max_val)[0]

        # Randomly select one of these indices
        return np.random.choice(max_indices)

    return np.apply_along_axis(random_choice_of_max, 1, arr)


def get_discriminator(model_type, use_soft_labels, n_class=2, **params):
    if model_type == 'logistic':
        return LogReg(use_soft_labels, n_class=n_class, **params)
    elif model_type == "knn":
        return KNN(use_soft_labels, n_class=n_class, **params)
    elif model_type == "mlp":
        return MLP(use_soft_labels, n_class=n_class, **params)
    else:
        raise ValueError('discriminator disc_model not supported.')


def train_disc_model(model_type, distance_metric, xs_tr, ys_tr_soft, valid_dataset, use_soft_labels, tune_end_model, tune_metric):
    # prepare discriminator
    if model_type == "knn":
        disc_model = get_discriminator(model_type, use_soft_labels, n_class=valid_dataset.n_class, metric=distance_metric)
    else:
        disc_model = get_discriminator(model_type, use_soft_labels, n_class=valid_dataset.n_class)

    if use_soft_labels:
        ys_tr = ys_tr_soft
    else:
        ys_tr = random_argmax(ys_tr_soft)
        if np.min(ys_tr) == np.max(ys_tr):
            warnings.warn("All labels are the same, no need to train a discriminator.")
            return None

    if tune_end_model:
        disc_model.tune_params(xs_tr, ys_tr, valid_dataset.features, valid_dataset.labels, tune_metric=tune_metric)

    disc_model.fit(xs_tr, ys_tr)
    return disc_model


def train_resnet(model, train_loader, valid_loader, n_epochs, lr, weight_decay, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_valid_loss = float('inf')
    best_model = None
    patience = 5
    early_stopping_counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                valid_loss += loss.item()

        scheduler.step()

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping")
            break

        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    model.load_state_dict(best_model)
    return model


class EndClassifier:
    """Downstream Classifier backbone
    """
    def __init__(self, use_soft_labels, n_class=2, **kwargs):
        self.use_soft_labels = use_soft_labels
        self.n_class = n_class
        self.kwargs = kwargs
        self.model = None

    def tune_params(self, x_train, y_train, x_valid, y_valid, tune_metric='acc'):
        raise NotImplementedError

    def fit(self, xs, ys):
        raise NotImplementedError

    def predict_proba(self, xs):
        raise NotImplementedError

    def predict(self, xs):
        proba = self.predict_proba(xs)
        return random_argmax(proba)


class SoftLabelKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=10, weights="uniform", metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    def fit(self, X, y_soft):
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric)
        self.knn.fit(X, np.argmax(y_soft, axis=1))
        self.y_train_ = y_soft
        self.classes_ = np.arange(y_soft.shape[1])
        return self

    def predict_proba(self, X):
        distances, indices = self.knn.kneighbors(X)
        if self.weights == "uniform":
            weights = np.ones((X.shape[0], self.n_neighbors))
        elif self.weights == "distance":
            weights = 1 / (distances + 1e-10)  # Avoid division by zero
        else:
            raise ValueError("weights not recognized: {}".format(self.weights))
        weighted_votes = np.zeros((X.shape[0], len(self.classes_)))
        for i, neigh_idxs in enumerate(indices):
            weighted_votes[i, :] = np.sum(self.y_train_[neigh_idxs,:] * weights[i, :].reshape(-1, 1), axis=0)
        return weighted_votes / weighted_votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        return random_argmax(proba)


class SoftLabelLogRegClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, solver='liblinear', max_iter=1000, class_weight="balanced"):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.logreg = LogisticRegression(C=C, solver=solver, max_iter=max_iter, class_weight=class_weight)
        self.y_unique = None  # record the classes with positive weights
        self.n_class = None

    def fit(self, X, y_soft):
        self.n_class = y_soft.shape[1]
        # Expand data based on soft labels
        X_expanded = np.repeat(X, self.n_class, axis=0)  # Duplicate each row
        y_expanded = np.array(list(range(self.n_class)) * len(y_soft))  # Alternate 0 and 1 for each duplicate
        sample_weight = y_soft.flatten()  # Flatten the soft labels to use as weights

        positive_weight_idx = np.where(sample_weight > 0)[0]
        X_expanded = X_expanded[positive_weight_idx]
        y_expanded = y_expanded[positive_weight_idx]
        sample_weight = sample_weight[positive_weight_idx]

        # Train the disc_model using the expanded dataset and sample weights
        self.y_unique = np.unique(y_expanded)
        if len(self.y_unique) == 1:
            warnings.warn("All labels are the same, no need to train a discriminator.")
            return None

        self.logreg = LogisticRegression(C=self.C, solver=self.solver, max_iter=self.max_iter, class_weight=self.class_weight)
        self.logreg.fit(X_expanded, y_expanded, sample_weight=sample_weight)

    def predict(self, X):
        if len(self.y_unique) == 1:
            return np.ones(X.shape[0]) * self.y_unique[0]
        else:
            return self.logreg.predict(X)

    def predict_proba(self, X):
        if len(self.y_unique) == 1:
            proba = np.zeros((X.shape[0], self.n_class))
            proba[:, self.y_unique[0]] = 1.0
            return proba
        else:
            proba = np.zeros((X.shape[0], self.n_class))
            proba[:, self.y_unique] = self.logreg.predict_proba(X)
            return proba


class KNN(EndClassifier):
    def __init__(self, use_soft_labels, n_class=2, **kwargs):
        super().__init__(use_soft_labels, n_class, **kwargs)
        if use_soft_labels:
            self.model = SoftLabelKNNClassifier(**kwargs)
        else:
            self.model = KNeighborsClassifier(**kwargs)
        self.best_params = {}

    def tune_params(self, x_train, y_train, x_valid, y_valid, tune_metric='acc'):
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 10),
                # 'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine']),
                # 'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            }
            self.model.set_params(**params)
            self.model.fit(x_train, y_train)
            preds = self.model.predict(x_valid)
            if tune_metric == 'acc':
                return accuracy_score(y_valid, preds)
            elif tune_metric == 'f1':
                if self.n_class == 2:
                    return f1_score(y_valid, preds)
                else:
                    return f1_score(y_valid, preds, average='macro')
            else:
                raise ValueError('tune metric not supported.')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        self.best_params = study.best_params
        self.model.set_params(**self.best_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, xs):
        proba = self.predict_proba(xs)
        return random_argmax(proba)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LogReg(EndClassifier):
    def __init__(self, use_soft_labels, n_class=2, **kwargs):
        super().__init__(use_soft_labels, n_class, **kwargs)
        if use_soft_labels:
            self.model = SoftLabelLogRegClassifier(**kwargs)
        else:
            self.model = LogisticRegression(**kwargs)
        self.best_params = {}

    def tune_params(self, x_train, y_train, x_valid, y_valid, tune_metric='acc'):
        def objective(trial):
            params = {
                'C': trial.suggest_categorical('C', [0.001, 0.01, 0.1, 1, 10, 100]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
                'solver': trial.suggest_categorical('solver', ['liblinear']),
                'max_iter': trial.suggest_categorical('max_iter', [1000]),
            }
            self.model.set_params(**params)
            self.model.fit(x_train, y_train)
            preds = self.model.predict(x_valid)
            if tune_metric == 'acc':
                return accuracy_score(y_valid, preds)
            elif tune_metric == 'f1':
                if self.n_class == 2:
                    return f1_score(y_valid, preds)
                else:
                    return f1_score(y_valid, preds, average='macro')
            else:
                raise ValueError('tune metric not supported.')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        self.best_params = study.best_params
        self.model.set_params(**self.best_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        proba = self.predict_proba(X)
        return random_argmax(proba)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MLP(EndClassifier):
    def __init__(self, use_soft_labels, n_class=2, **kwargs):
        super().__init__(use_soft_labels, n_class, **kwargs)
        if use_soft_labels:
            self.model = MLPRegressor(**kwargs)
        else:
            self.model = MLPClassifier(**kwargs)
        self.best_params = {}

    def tune_params(self, x_train, y_train, x_valid, y_valid, tune_metric='acc'):
        def objective(trial):
            params = {
                'activation': trial.suggest_categorical('activation', ['relu']),
                'solver': trial.suggest_categorical('solver', ['adam']),
                'alpha': trial.suggest_categorical('alpha', [0.0001, 0.001, 0.01, 0.1]),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'learning_rate_init': trial.suggest_categorical('learning_rate_init', [0.001, 0.01, 0.1]),
                'max_iter': trial.suggest_categorical('max_iter', [1000]),
            }
            self.model.set_params(**params)
            self.model.fit(x_train, y_train)
            preds = self.model.predict(x_valid)
            if self.use_soft_labels:
                preds = preds.argmax(axis=1)

            if tune_metric == 'acc':
                return accuracy_score(y_valid, preds)
            elif tune_metric == 'f1':
                if self.n_class == 2:
                    return f1_score(y_valid, preds)
                else:
                    return f1_score(y_valid, preds, average='macro')
            else:
                raise ValueError('tune metric not supported.')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        self.best_params = study.best_params
        self.model.set_params(**self.best_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        proba = self.predict_proba(X)
        return random_argmax(proba)

    def predict_proba(self, X):
        if self.use_soft_labels:
            proba = self.model.predict(X)
            proba = proba / np.sum(proba, axis=1, keepdims=True)
            return proba
        else:
            return self.model.predict_proba(X)