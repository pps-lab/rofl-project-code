import unittest
import numpy as np

from src.aggregation.aggregators import TrimmedMean


class TrimmedMeanTest(unittest.TestCase):
    def test_aggregates_properly(self):

        w1 = np.array(((1, 5), (1, 5)))
        w2 = np.array(((2, 3), (2, 3)))
        w3 = np.array(((10, 11), (10, 11)))

        average = TrimmedMean(0.34, 1.0).aggregate(np.zeros(w1.shape), [w1, w2, w3])
        print(average)


if __name__ == '__main__':
    unittest.main()
