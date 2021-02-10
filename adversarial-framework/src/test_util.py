from unittest import TestCase
import numpy as np
import util

class UtilityTest(TestCase):

    def test_aggregate_weights_masked_simple(self):
        current_weights = [VariableWrapper([0.]) for _ in range(8)]

        w1 = np.array([[1], [1], [1], [1], [2], [2], [2], [2]])
        w2 = np.array([[3], [3], [3], [3], [4], [4], [4], [4]])
        w3 = np.array([[5], [5], [5], [5], [6], [6], [6], [6]])
        w1m = np.array([[True], [True], [True], [False], [True], [True], [True], [False]])
        w2m = np.array([[False], [True], [True], [True], [False], [True], [True], [True]])
        w3m = np.array([[False], [True], [False], [True], [False], [True], [False], [True]])

        res = util.aggregate_weights_masked(current_weights, 1.0, 3.0, 1.0, [w1m, w2m, w3m], [w1, w2, w3])
        np.testing.assert_array_almost_equal(np.array([
            [1./3.], [3], [1.3333], [2.666667], [2./3.], [4], [2], [3.33333]
        ]), res, decimal=3)

    def test_aggregate_weights_masked_emptynan(self):
        current_weights = [VariableWrapper([5.])] + [VariableWrapper([0.]) for _ in range(7)]

        w1 = np.array([[1], [1], [1], [1], [2], [2], [2], [2]])
        w2 = np.array([[3], [3], [3], [3], [4], [4], [4], [4]])
        w3 = np.array([[5], [5], [5], [5], [6], [6], [6], [6]])
        w1m = np.array([[False], [True], [True], [False], [True], [True], [True], [False]])
        w2m = np.array([[False], [True], [True], [True], [False], [True], [True], [True]])
        w3m = np.array([[False], [True], [False], [True], [False], [True], [False], [True]])

        res = util.aggregate_weights_masked(current_weights, 1.0, 3.0, 1.0, [w1m, w2m, w3m], [w1, w2, w3])
        np.testing.assert_array_almost_equal(np.array([
            [5], [3], [1.33333], [2.6666667], [2./3.], [4.], [2.], [3.33333]
        ]), res, decimal=3)

    def test_aggregate_weights_masked_multidim(self):
        current_weights = [VariableWrapper([2., 2.]), VariableWrapper([2., 2.])]

        w1 = np.array([[1, 1], [1, 1]])
        w2 = np.array([[3, 3], [3, 3]])
        w3 = np.array([[5, 5], [5, 5]])
        w1m = np.array([[False, False], [True, True]])
        w2m = np.array([[True, False], [True, False]])
        w3m = np.array([[False, True], [False, True]])

        res = util.aggregate_weights_masked(current_weights, 1.0, 3.0, 1.0, [w1m, w2m, w3m], [w1, w2, w3])
        np.testing.assert_array_almost_equal(np.array([
            [2.33333, 3], [2, 2.66667]
        ]), res, decimal=3)


class VariableWrapper:
    """Wrapper to wrap variable in to provide compatibility with Tensorflow"""
    def __init__(self, var):
        self.var = var

    def __add__(self, o):
        return self.var + o

    def numpy(self):
        return self.var

