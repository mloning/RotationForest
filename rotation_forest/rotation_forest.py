#!/usr/bin/env python3 -u
# coding: utf-8

# code from https://github.com/joshloyal/RotationForest under MIT license

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble.forest import ForestClassifier
from sklearn.ensemble.base import _set_random_states
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_random_state, check_is_fitted
from sklearn.exceptions import DataConversionWarning
from warnings import warn

from itertools import islice


class RotationTreeClassifier(DecisionTreeClassifier):
    def __init__(self,
                 max_features_subset=3,
                 min_features_subset=3,
                 p_sample_subset=0.5,
                 bootstrap_sample_subset=False,
                 transformation="pca",
                 criterion="entropy",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
                 presort=False):

        self.max_features_subset = max_features_subset
        self.min_features_subset = min_features_subset
        self.p_sample_subset = p_sample_subset
        self.transformation = transformation
        self.bootstrap_sample_subset = bootstrap_sample_subset

        super(RotationTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            presort=presort)

        # set in init
        self.n_samples_ = None
        self.n_features_ = None
        self.classes_ = None
        self.n_outputs_ = None
        self._rng = None
        self.base_transformer_ = None

    def transform(self, X, y=None):
        check_is_fitted(self, "rotation_matrix_")
        return np.dot(X, self.rotation_matrix_)

    def _make_transformer(self, random_state=None):
        """Make and configure a copy of the `base_transformer` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        transformer = clone(self.base_transformer_)

        if random_state is not None:
            _set_random_states(transformer, random_state)

        return transformer

    def _fit_transfomers(self, X, y):
        self.rotation_matrix_ = np.zeros((self.n_features_, self.n_features_), dtype=np.float32)
        self.feature_subsets_ = self._random_feature_subsets(min_length=self.min_features_subset,
                                                             max_length=self.max_features_subset)

        for i, feature_subset in enumerate(self.feature_subsets_):
            sample_subset = self._random_sample_subset(y, bootstrap=self.bootstrap_sample_subset)

            # add more samples if less samples than features in subset
            while len(sample_subset) < len(feature_subset):
                n_new_samples = len(feature_subset) - len(sample_subset)
                new_sample_subset = self._random_sample_subset(y, n_samples=n_new_samples)
                sample_subset = np.vstack([sample_subset, new_sample_subset])

            n_attempts = 0
            while n_attempts < 10:
                pca = self._make_transformer(random_state=self.random_state)

                with np.errstate(divide='ignore', invalid='ignore'):
                    pca.fit(X[sample_subset, feature_subset])

                # check pca fit
                is_na = np.any(np.isnan(pca.explained_variance_ratio_))
                # is_inf = np.any(np.isinf(pca.explained_variance_ratio_))

                if is_na:
                    n_attempts += 1
                    new_sample_subset = self._random_sample_subset(y, n_samples=10)
                    sample_subset = np.vstack([sample_subset, new_sample_subset])

                else:
                    self.rotation_matrix_[np.ix_(feature_subset, feature_subset)] = pca.components_
                    break

    def _random_sample_subset(self, y, n_samples=None, bootstrap=False):
        """Select subset of samples (with replacements) conditional on random subset of classes"""
        # get random state object
        rng = self._rng

        # get random subset of classes if not given
        n_classes = rng.randint(1, len(self.classes_) + 1)
        classes = rng.choice(self.classes_, size=n_classes, replace=False)

        # get samples for selected classes
        isin_classes = np.where(np.isin(y, classes))[0]
        n_isin_classes = len(isin_classes)

        # set number of samples in subset
        if n_samples is None:
            n_samples = np.int(np.ceil(n_isin_classes * self.p_sample_subset))
        # if n_samples is given, ensure is less than the number of samples in the selected classes
        else:
            if n_samples > n_isin_classes:
                n_samples = n_isin_classes

        # randomly select subset of samples for selected classes
        sample_subset = rng.choice(isin_classes, size=n_samples, replace=bootstrap)
        return sample_subset[:, None]

    def _random_feature_subsets(self, min_length, max_length):
        """Randomly select subsets of features"""
        # get random state object
        rng = self._rng

        # shuffle features
        features = np.arange(self.n_features_)
        rng.shuffle(features)

        # if length is not variable, use available function to split into equally sized arrays
        if min_length == max_length:
            n_subsets = self.n_features_ // max_length
            return np.array_split(features, n_subsets)

        # otherwise iterate through features, selecting uniformly random number of features within
        # given bounds for each subset
        subsets = []
        it = iter(features)  # iterator over features
        while True:
            # draw random number of features within bounds
            n_features_in_subset = rng.random.randint(min_length, max_length + 1)

            # select number of features and move iterator ahead
            subset = list(islice(it, n_features_in_subset))

            # append if non-empty, otherwise break while loop
            if len(subset) > 0:
                subsets.append(np.array(subset))
            else:
                break

        # subsets = self._random_feature_subset_rand_n(features, min_length, max_length)
        return subsets

    def fit(self, X, y, **kwargs):
        # check inputs
        X, y = check_X_y(X, y)

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.transformation == "pca":
            self.base_transformer_ = PCA()
        elif self.transformation == "randomized":
            self.base_transformer_ = PCA(svd_solver="randomized")
        else:
            raise ValueError("`transformation` must be either 'pca' or 'randomized'.")

        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_outputs_ = y.shape[1]
        self._rng = check_random_state(self.random_state)

        # fit transfomers
        self._fit_transfomers(X, y)

        # transform data
        Xt = self.transform(X)

        # fit estimators on transformed data
        super(RotationTreeClassifier, self).fit(Xt, y, **kwargs)

    def predict_proba(self, X, check_input=True):
        check_is_fitted(self, 'rotation_matrix_')

        # check input
        X = check_array(X)

        Xt = self.transform(X)
        return super(RotationTreeClassifier, self).predict_proba(Xt, check_input)

    def predict(self, X, check_input=True):
        Xt = self.transform(X)
        return super(RotationTreeClassifier, self).predict(Xt, check_input)

    def apply(self, X, check_input=True):
        check_is_fitted(self, 'rotation_matrix_')

        # check input
        X = check_array(X)

        Xt = self.transform(X)
        return super(RotationTreeClassifier, self).apply(Xt, check_input)

    def decision_path(self, X, check_input=True):
        check_is_fitted(self, 'rotation_matrix_')

        # check input
        X = check_array(X)

        Xt = self.transform(X)
        return super(RotationTreeClassifier, self).decision_path(Xt, check_input)

    @property
    def feature_importances_(self):
        raise NotImplementedError()


class RotationForestClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=200,
                 max_features_subset=3,
                 min_features_subset=3,
                 p_sample_subset=0.5,
                 bootstrap_sample_subset=False,
                 transformation="pca",
                 criterion="entropy",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 max_leaf_nodes=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):

        super(RotationForestClassifier, self).__init__(
            base_estimator=RotationTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=["min_features_subset", "max_features_subset",
                              "p_sample_subset", "bootstrap_sample_subset", "transformation",
                              "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"],
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.min_features_subset = min_features_subset
        self.max_features_subset = max_features_subset
        self.p_sample_subset = p_sample_subset
        self.bootstrap_sample_subset = bootstrap_sample_subset
        self.transformation = transformation
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
