import unittest
import numpy as np
from scipy.spatial.distance import euclidean, hamming

class TestDistanceFunctions(unittest.TestCase):
    
    def chi_square_distance(self, histA, histB, eps=1e-10):
        return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

    def test_chi_square_distance(self):
        histA = np.array([5, 4, 2, 3])
        histB = np.array([1, 2, 4, 2])
        expected = 2.0999999999646666  # expected result
        result = self.chi_square_distance(histA, histB)
        self.assertEqual(result, expected)

    def test_euclidean_distance(self):
        uploaded_feature = np.array([1, 2, 3])
        feature = np.array([4, 5, 6])
        expected = 5.196152422706632  # expected result
        result = euclidean(uploaded_feature, feature)
        self.assertEqual(result, expected)

    def test_hamming_distance(self):
        uploaded_feature = np.array([1, 1, 1, 0, 0, 1])
        feature = np.array([1, 1, 1, 0, 1, 1])
        expected = 1  # expected result (pre-calculated)
        result = hamming(uploaded_feature, feature) * len(feature)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()