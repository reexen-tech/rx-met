"""Regression tests for histogram calibration against independent bin counts."""
import unittest

import numpy as np
import torch

from aimet_torch.v2.quantization.encoding_analyzer import (
    PercentileEncodingAnalyzer, SqnrEncodingAnalyzer,
)


class HistogramCalibrationTest(unittest.TestCase):
    def test_large_constant_then_new_extreme(self):
        for sign in (1, -1):
            with self.subTest(sign=sign):
                analyzer = PercentileEncodingAnalyzer((), num_bins=8, percentile=90)
                first = torch.full((4,), sign * 1e20)
                later = first[:1] - 5e19
                analyzer.update_stats(first)
                initial = analyzer.observer.get_stats()[0]
                self.assertTrue((initial.bin_edges.diff() > 0).all())
                analyzer.update_stats(later)
                stat = analyzer.observer.get_stats()[0]
                self.assertEqual(stat.histogram.tolist(), [1, 0, 0, 0, 0, 0, 0, 4])
                low, high = analyzer.compute_encodings(65535, False)
                self.assertLessEqual(low.item(), later.item())
                self.assertGreaterEqual(high.item(), first[0].item())

    def test_first_mixed_nonfinite_batch_keeps_finite_growth_limits(self):
        analyzer = SqnrEncodingAnalyzer((), num_bins=4, histogram_growth_limit=2)
        analyzer.update_stats(torch.tensor([0., 2., float("nan"), float("inf"), -float("inf")]))
        analyzer.update_stats(torch.tensor([-10., 10.]))
        stats = analyzer.observer.get_stats()[0]
        self.assertEqual(stats.bin_edges.tolist(), [-2., -0.5, 1., 2.5, 4.])
        self.assertEqual(stats.histogram.tolist(), [1, 2, 2, 1])

    def test_no_finite_samples_in_one_channel_preserves_prior_statistics(self):
        analyzer = PercentileEncodingAnalyzer((2, 1), num_bins=4)
        analyzer.update_stats(torch.tensor([[1., 2.], [-3., -2.]]))
        with self.assertRaisesRegex(RuntimeError, "finite"):
            analyzer.update_stats(torch.tensor([[5., 6.], [float("inf"), float("nan")]]))
        analyzer.update_stats(torch.tensor([[2., 1.], [-3., -2.]]))
        self.assertEqual([s.histogram.tolist() for s in analyzer.observer.get_stats()],
                         [[2, 0, 0, 2], [2, 0, 0, 2]])

    def test_percentile_encloses_retained_one_sided_bins(self):
        for sign in (1, -1):
            values = torch.tensor([0., 1., 1., 1., 1.]) * sign
            analyzer = PercentileEncodingAnalyzer((), num_bins=2, percentile=90)
            analyzer.update_stats(values)
            low, high = analyzer.compute_encodings(255, False)
            self.assertEqual((low.item(), high.item()), (values.min(), values.max()))

    def test_large_bin_keeps_single_later_observation(self):
        analyzer = PercentileEncodingAnalyzer((), num_bins=2)
        values = torch.zeros(2**20)
        for _ in range(16):
            analyzer.update_stats(values)
        analyzer.update_stats(torch.zeros(1))
        self.assertEqual(analyzer.observer.get_stats()[0].histogram.tolist(), [0, 2**24 + 1])

    def test_one_sided_narrow_histogram_preserves_counts_on_growth(self):
        # Neighboring representable float32 values separated by 1/16.
        analyzer = SqnrEncodingAnalyzer((), num_bins=8)
        analyzer.update_stats(torch.tensor([1e6, 1e6 + 0.0625]))
        analyzer.update_stats(torch.tensor([1e6, 1e6 + 0.125]))
        self.assertEqual(analyzer.observer.get_stats()[0].histogram.tolist(), [2, 0, 0, 1, 0, 0, 0, 1])

    def test_histogram_interior_edges_belong_to_the_right_bin(self):
        analyzer = PercentileEncodingAnalyzer((), num_bins=10)
        # Include the float32 numbers immediately to either side of an ideal edge.
        edges = np.linspace(-3., 7., 11)
        inner = edges[1:-1].astype(np.float32)
        data = np.concatenate((edges[[0, -1]], inner,
                               np.nextafter(inner, -np.inf), np.nextafter(inner, np.inf))).astype(np.float32)
        analyzer.update_stats(torch.from_numpy(data))
        expected, _ = np.histogram(data.astype(np.float64), edges)
        self.assertEqual(analyzer.observer.get_stats()[0].histogram.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
