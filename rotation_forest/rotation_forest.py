#!/usr/bin/env python3 -u
# coding: utf-8

# original basic implementation from https://github.com/joshloyal/RotationForest under MIT license

# original publication
# @article{rodriguez2006rotation,
#   title={Rotation forest: A new classifier ensemble method},
#   author={Rodriguez, Juan Jos{\'e} and Kuncheva, Ludmila I and Alonso, Carlos J},
#   journal={IEEE transactions on pattern analysis and machine intelligence},
#   volume={28},
#   number={10},
#   pages={1619--1630},
#   year={2006},
#   publisher={IEEE}
# }

__author__ = ["Markus Löning"]
__all__ = [
    "RotationTreeClassifier",
    "RotationForestRegressor",
    "RotationForestClassifier",
    "RotationTreeRegressor"
]

from itertools import islice

import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble.base import _set_random_states
from sklearn.ensemble.forest import ForestClassifier, ForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_random_state, check_is_fitted


class BaseRotationTree:

    def transform(self, X, y=None):
        check_is_fitted(self, "rotation_matrix_")
        return np.dot(X, self.rotation_matrix_)

    def _make_transformer(self, random_state=None):
        """Make and configure a copy of the `base_transformer` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        transformer = clone(self.base_transformer)

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
            # make sure subsets are of given length by reusing some of the features
            mod = self.n_features_ % n_subsets
            if mod > 0:
                n_extra_features = min_length - mod
                n_subsets = n_subsets + 1
                extra_features = rng.choice(features, size=n_extra_features, replace=False)
                features = np.hstack([features, extra_features])
            return np.array_split(features, n_subsets)

        # otherwise iterate through features, selecting uniformly random number of features within
        # given bounds for each subset
        subsets = []
        it = iter(features)  # iterator over features
        while True:
            # draw random number of features within bounds
            n_features_in_subset = rng.randint(min_length, max_length + 1)

            # select number of features and move iterator ahead
            subset = list(islice(it, n_features_in_subset))

            # append subsets, for last subset, check if there are enough features,
            # otherwise randomly reuse some of them, finally break while loop
            len_subset = len(subset)
            if min_length <= len_subset <= max_length:
                subsets.append(np.array(subset))
            elif 0 < len_subset < min_length:
                n_extra_features = min_length - len_subset
                extra_features = rng.choice(features, size=n_extra_features, replace=False)
                subset = np.hstack([subset, extra_features])
                subsets.append(subset)
            else:
                break

        return subsets


class RotationTreeClassifier(DecisionTreeClassifier, BaseRotationTree):

    def __init__(self,
                 max_features_subset=3,
                 min_features_subset=3,
                 p_sample_subset=0.5,
                 bootstrap_sample_subset=False,
                 criterion="gini",
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

        if not (isinstance(min_features_subset, int) and min_features_subset > 0):
            raise ValueError("min_features_subset must be a positive (> 0) integer.")

        if not (isinstance(max_features_subset, int) and max_features_subset > 0):
            raise ValueError("min_features_subset must be a positive (> 0) integer.")

        if min_features_subset > max_features_subset:
            raise ValueError("min_features_subset must be smaller than or "
                             "equal to max_features_subset")

        if not 0 < p_sample_subset <= 1:
            raise ValueError("p_sample_subset must be > 0 and <= 1.")

        self.max_features_subset = max_features_subset
        self.min_features_subset = min_features_subset
        self.p_sample_subset = p_sample_subset
        self.bootstrap_sample_subset = bootstrap_sample_subset
        self.base_transformer = PCA()

        # set in init
        self.n_samples_ = None
        self.n_features_ = None
        self.classes_ = None
        self.n_outputs_ = None
        self._rng = None

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

    def fit(self, X, y, check_input=True, **kwargs):
        self._rng = check_random_state(self.random_state)
        X, y = check_X_y(X, y, multi_output=True)

        self.n_samples_, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        self.classes_ = np.unique(y)

        # fit transfomers
        self._fit_transfomers(X, y)

        # transform data
        Xt = self.transform(X)

        # fit estimators on transformed data
        super(RotationTreeClassifier, self).fit(Xt, y, check_input=check_input, **kwargs)

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


class RotationForestClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=200,
                 max_features_subset=3,
                 min_features_subset=3,
                 p_sample_subset=0.5,
                 bootstrap_sample_subset=False,
                 criterion="gini",
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
                              "p_sample_subset", "bootstrap_sample_subset",
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
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state


class RotationTreeRegressor(DecisionTreeRegressor, BaseRotationTree):

    def __init__(self,
                 max_features_subset=3,
                 min_features_subset=3,
                 p_sample_subset=0.5,
                 bootstrap_sample_subset=False,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 presort=False):

        if not (isinstance(min_features_subset, int) and min_features_subset > 0):
            raise ValueError("min_features_subset must be a positive (> 0) integer.")

        if not (isinstance(max_features_subset, int) and max_features_subset > 0):
            raise ValueError("min_features_subset must be a positive (> 0) integer.")

        if min_features_subset > max_features_subset:
            raise ValueError("min_features_subset must be smaller than or "
                             "equal to max_features_subset")

        if not 0 < p_sample_subset <= 1:
            raise ValueError("p_sample_subset must be > 0 and <= 1.")

        self.max_features_subset = max_features_subset
        self.min_features_subset = min_features_subset
        self.p_sample_subset = p_sample_subset
        self.bootstrap_sample_subset = bootstrap_sample_subset
        self.base_transformer = PCA()

        # set in init
        self.n_samples_ = None
        self.n_features_ = None
        self.classes_ = None
        self.n_outputs_ = None
        self._rng = None

        super(RotationTreeRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            presort=presort)

    def fit(self, X, y, check_input=True, **kwargs):
        self._rng = check_random_state(self.random_state)
        X, y = check_X_y(X, y)

        self.n_samples_, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        # fit transfomers
        self._fit_transfomers(X, y)

        # transform data
        Xt = self.transform(X)

        # fit estimators on transformed data
        super(RotationTreeRegressor, self).fit(Xt, y, check_input=check_input, **kwargs)

    def predict(self, X, check_input=True):
        Xt = self.transform(X)
        return super(RotationTreeRegressor, self).predict(Xt, check_input)

    def apply(self, X, check_input=True):
        check_is_fitted(self, 'rotation_matrix_')

        # check input
        X = check_array(X)

        Xt = self.transform(X)
        return super(RotationTreeRegressor, self).apply(Xt, check_input)

    def decision_path(self, X, check_input=True):
        check_is_fitted(self, 'rotation_matrix_')

        # check input
        X = check_array(X)

        Xt = self.transform(X)
        return super(RotationTreeRegressor, self).decision_path(Xt, check_input)

    def _random_sample_subset(self, n_samples=None, bootstrap=False):
        """Select subset of instances (with replacements) conditional on random subset of classes"""
        # get random state object
        rng = self._rng

        # instance index
        samples = np.arange(self.n_samples_)

        # set number of samples in subset
        if n_samples is None:
            n_samples = np.int(np.ceil(self.n_samples_ * self.p_sample_subset))
        # if n_samples is given, ensure is less than the number of samples in the selected classes
        else:
            if n_samples > self.n_samples_:
                n_samples = self.n_samples_

        # randomly select bootstrap subset of instances for selected classes
        samples_subset = rng.choice(samples, size=n_samples, replace=bootstrap)
        return samples_subset[:, None]

    @property
    def feature_importances_(self):
        raise NotImplementedError()


class RotationForestRegressor(ForestRegressor):
    def __init__(self,
                 n_estimators=200,
                 max_features_subset=3,
                 min_features_subset=3,
                 p_sample_subset=0.5,
                 bootstrap_sample_subset=False,
                 criterion="mse",
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
                 warm_start=False):
        super(RotationForestRegressor, self).__init__(
            base_estimator=RotationTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=["min_features_subset", "max_features_subset",
                              "p_sample_subset", "bootstrap_sample_subset",
                              "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"],
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.min_features_subset = min_features_subset
        self.max_features_subset = max_features_subset
        self.p_sample_subset = p_sample_subset
        self.bootstrap_sample_subset = bootstrap_sample_subset
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
