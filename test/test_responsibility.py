import unittest

import numpy as np

import sati


class TestResponsibility(unittest.TestCase):
    """Tests for sati.responsibility.Responsibility."""

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            sati.Responsibility(np.zeros((2, 2, 2)), 2)

        with self.assertRaises(ValueError):
            sati.Responsibility(np.zeros((8, 8)), 0)

    def test_classify_threshold(self):
        image = np.zeros((2, 3), dtype=float)
        rsp = sati.Responsibility(image, 3)

        rsp.values[0] = np.array([[0.2, 0.0, 0.0], [0.0, 0.1, 0.0]])
        rsp.values[1] = np.array([[0.1, 0.7, 0.0], [0.0, 0.0, 0.0]])
        rsp.values[2] = np.array([[0.0, 0.1, 0.4], [0.6, 0.0, 0.0]])

        label = rsp.classify(threshold=0.5)
        expected = np.array([[np.nan, 1.0, np.nan], [2.0, np.nan, np.nan]])
        np.testing.assert_equal(label, expected)

    def test_initial_guess_two_terraces(self):
        image = np.zeros((32, 32), dtype=float)
        image[16:, :] = 1.0

        rsp = sati.Responsibility(image, 2)
        rsp.initial_guess(
            tolerance=0.1,
            stride=4,
            min_flood_area=4,
            min_candidate_area=20,
            min_edge_confidence=0.0,
            adjacency_gap=2.0,
        )

        self.assertEqual(rsp.values.shape, (2, 32, 32))
        self.assertEqual(rsp.initial_candidates.shape, (32, 32))
        self.assertGreater(np.count_nonzero(rsp.initial_candidates), 0)

        # Each classified pixel should be one-hot across class axis.
        sum_per_pixel = rsp.values.sum(axis=0)
        self.assertTrue(np.all((sum_per_pixel == 0.0) | (sum_per_pixel == 1.0)))

        label = rsp.classify(threshold=0.5)
        top = label[:16, :]
        bottom = label[16:, :]

        top_assigned = top[~np.isnan(top)]
        bottom_assigned = bottom[~np.isnan(bottom)]

        self.assertGreater(top_assigned.size, 0)
        self.assertGreater(bottom_assigned.size, 0)

        top_majority = np.bincount(top_assigned.astype(int)).argmax()
        bottom_majority = np.bincount(bottom_assigned.astype(int)).argmax()

        self.assertNotEqual(top_majority, bottom_majority)

    def test_initial_guess_argument_validation(self):
        image = np.zeros((16, 16), dtype=float)
        rsp = sati.Responsibility(image, 2)

        cases = [
            {"tolerance": 0.0},
            {"tolerance": -0.1},
            {"tolerance": 0.1, "stride": 0},
            {"tolerance": 0.1, "min_flood_area": 0},
            {"tolerance": 0.1, "min_overlap_pixels": 0},
            {"tolerance": 0.1, "min_overlap_fraction": -0.1},
            {"tolerance": 0.1, "min_overlap_fraction": 1.1},
            {"tolerance": 0.1, "min_candidate_area": 0},
            {"tolerance": 0.1, "adjacency_gap": -1.0},
            {"tolerance": 0.1, "edge_band": -1.0},
            {"tolerance": 0.1, "min_edge_confidence": -1.0},
            {"tolerance": 0.1, "connectivity": 0},
            {"tolerance": 0.1, "connectivity": 3},
            {"tolerance": 0.1, "core_support_fraction": 0.0},
            {"tolerance": 0.1, "core_support_fraction": 1.1},
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    rsp.initial_guess(**kwargs)


if __name__ == "__main__":
    unittest.main()
