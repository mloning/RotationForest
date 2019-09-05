import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from itertools import combinations

from rotation_forest import RotationTreeClassifier, RotationForestClassifier


def make_classification_data(n_samples=500, n_features=30, redundant_size=0.1,
                             informative_size=0.6, repeated_size=0.1):
    assert (redundant_size + informative_size + repeated_size) < 1
    n_redundant = int(redundant_size * n_features)
    n_informative = int(informative_size * n_features)
    n_repeated = int(repeated_size * n_features)

    X, y = make_classification(n_samples=n_samples, n_features=n_features, flip_y=0.03,
                               n_informative=n_informative, n_redundant=n_redundant,
                               n_repeated=n_repeated, random_state=9)

    return X, y.ravel()


@pytest.mark.parametrize("estimator", [
    RotationTreeClassifier(),
    RotationForestClassifier(n_estimators=2)
])
def test_classifier_fit_predict(estimator):
    """simple test for fit and predict"""
    X, y = make_classification_data()
    xt, xv, yt, yv = train_test_split(X, y, test_size=0.3)
    estimator.fit(xt, yt)

    proba = estimator.predict_proba(xv)
    assert proba.shape[0] == xv.shape[0]
    assert np.all(proba <= 1)
    assert np.all(proba >= 0)

    yhat = estimator.predict(xv)
    assert yhat.shape[0] == xv.shape[0]
    np.testing.assert_array_equal(np.unique(yhat), np.array([0, 1]))


@pytest.mark.parametrize("estimator", [RotationTreeClassifier, RotationForestClassifier])
@pytest.mark.parametrize("bad_args", [
    {"min_features_subset": 3, "max_features_subset": 2}
])
def test_bad_input_args(estimator, bad_args):
    """check bad input args raise errors"""
    X, y = make_classification_data(n_samples=20, n_features=20)
    with pytest.raises(ValueError):
        estimator(**bad_args).fit(X, y)


@pytest.mark.parametrize("min_features_subset, max_features_subset", [
    (3, 3),
    (2, 5),
])
@pytest.mark.parametrize("n_features", [10, 20])
def test_random_feature_subsets(min_features_subset, max_features_subset, n_features):
    """check we generate disjoint feature subsets that cover all features"""
    X, y = make_classification_data(n_samples=10, n_features=n_features)

    # get random subsets
    estimator = RotationTreeClassifier(min_features_subset=min_features_subset,
                                       max_features_subset=max_features_subset)
    estimator.fit(X, y)
    subsets = estimator.feature_subsets_

    # check length
    assert all([min_features_subset <= len(subset) <= max_features_subset
                for subset in subsets])

    # check if all features have been used
    expected_features = np.arange(n_features)
    actual_features = np.sort(np.unique(np.hstack(subsets)))
    np.testing.assert_array_equal(expected_features, actual_features)

    # check if subsets are disjoint (except for the last which may reuse some features)
    for a, b in combinations(subsets[:-1], 2):
        a = set(a)
        assert not any(i in a for i in b)


def test_rotation_matrix():
    """Smoke test for rotation forest """
    n_features = 6
    X, y = make_classification_data(n_features=n_features)
    estimator = RotationTreeClassifier(min_features_subset=3, max_features_subset=3)
    estimator.fit(X, y)
    assert estimator.rotation_matrix_.shape == (n_features, n_features)

    # note that this random state generates the following subsets:
    subset1, subset2 = estimator.feature_subsets_

    # make sure the loadings are input in the proper order
    for feature in subset1:
        assert np.any(estimator.rotation_matrix_[:, feature][subset1] != 0)
        assert np.any(estimator.rotation_matrix_[:, feature][subset2] == 0)

    for feature in subset2:
        assert np.any(estimator.rotation_matrix_[:, feature][subset1] == 0)
        assert np.any(estimator.rotation_matrix_[:, feature][subset2] != 0)


def test_warm_start():
    """ Test if fitting incrementally with warm start gives a forest of the right
        size and the same results as a normal fit.
    """
    X, y = make_classification_data()
    clf_ws = None
    for n_estimators in [5, 10]:
        if clf_ws is None:
            clf_ws = RotationForestClassifier(n_estimators=n_estimators,
                                              random_state=1234,
                                              warm_start=True)
        else:
            clf_ws.set_params(n_estimators=n_estimators)
        clf_ws.fit(X, y)
        assert len(clf_ws) == n_estimators

    clf_no_ws = RotationForestClassifier(n_estimators=10,
                                         random_state=1234,
                                         warm_start=False)
    clf_no_ws.fit(X, y)
    assert set([tree.random_state for tree in clf_ws]) == set([tree.random_state for tree in clf_no_ws])

    np.testing.assert_array_equal(clf_ws.apply(X), clf_no_ws.apply(X))
