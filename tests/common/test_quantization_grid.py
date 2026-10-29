"""Framework-independent affine integer domains and source offset conversion."""

import unittest

import numpy as np

from aimet_common.quantization_grid import IntegerGrid, integer_bounds


class QuantizationGridTest(unittest.TestCase):
    def test_offset_conversions_preserve_integer_codes_and_real_values(self):
        for bits in (8, 16, 32):
            for signed in (True, False):
                with self.subTest(bits=bits, signed=signed):
                    points = [-7, 0] if signed else [7, 0]
                    grid = IntegerGrid([.125, .25], points, bits, signed)
                    for index_domain in (True, False):
                        offset = grid.additive_offset(index_domain=index_domain)
                        restored = IntegerGrid.from_offset(grid.scale, offset, bits, signed,
                                                          index_domain=index_domain)
                        self.assertEqual(restored.zero_point, points)
                        np.testing.assert_array_equal(restored.real_min, grid.real_min)
                        np.testing.assert_array_equal(restored.real_max, grid.real_max)
                        for code in (grid.qmin, 0, grid.qmax):
                            source_code = code-grid.qmin if index_domain else code
                            np.testing.assert_array_equal(
                                (source_code+offset)*grid.scale,
                                (code-np.asarray(points))*grid.scale)

    def test_dtype_is_independent_of_symmetry(self):
        for signed in (True, False):
            for symmetric in (True, False):
                record = dict(bitwidth=8, dtype='INT8' if signed else 'UINT8',
                              is_symmetric=str(symmetric), scale=.125, zero_point=0)
                grid = IntegerGrid.from_compiler_record(record)
                self.assertEqual(grid.signed, signed)
                self.assertEqual((grid.qmin, grid.qmax), integer_bounds(8, signed))

    def test_narrow_ranges_are_preserved_and_cannot_exceed_storage_type(self):
        grid = IntegerGrid(.125, 0, 8, True, -127, 127)
        self.assertEqual((grid.real_min, grid.real_max), (-15.875, 15.875))
        with self.assertRaisesRegex(ValueError, 'range'):
            IntegerGrid(.125, 0, 8, True, -129, 127)

    def test_invalid_grids_are_rejected(self):
        for scale, point in ((0, 0), (float('nan'), 0), (.125, .5), (.125, 128)):
            with self.subTest(scale=scale, point=point), self.assertRaises(ValueError):
                IntegerGrid(scale, point, 8, True)
        with self.assertRaisesRegex(ValueError, 'shape'):
            IntegerGrid([.125, .25], [0, 0, 0], 8, True)
        for signed in (None, 1, 'true'):
            with self.assertRaisesRegex(ValueError, 'signed'):
                integer_bounds(8, signed)

    def test_unsigned_numpy_zero_points_do_not_wrap_offsets(self):
        grid = IntegerGrid(.125, np.array(7, dtype=np.uint8), 8, False)
        self.assertEqual(grid.additive_offset().item(), -7)
        self.assertEqual(grid.real_min, -.875)
        restored = IntegerGrid.from_offset(.125, np.array(7, dtype=np.uint8), 8, True)
        self.assertEqual(restored.zero_point, -7)

    def test_legacy_protocols_are_resolved_by_saved_grid_constraints(self):
        for signed in (True, False):
            grid = IntegerGrid([.125], [7], 8, signed)
            for index_domain in (True, False):
                record = dict(bitwidth=8, scale=.125,
                              offset=grid.additive_offset(index_domain=index_domain).item(),
                              min=grid.real_min.item(), max=grid.real_max.item())
                restored = IntegerGrid.from_aimet_legacy_records([record], signed=signed)
                self.assertEqual(restored.zero_point, [7])
                record['max'] += 1.
                with self.assertRaisesRegex(ValueError, 'disagree'):
                    IntegerGrid.from_aimet_legacy_records([record], signed=signed)


if __name__ == '__main__':
    unittest.main()
