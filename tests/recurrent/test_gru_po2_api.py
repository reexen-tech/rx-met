"""GRU public PO2 names retain legacy configuration and checkpoint compatibility."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quant_gru import QuantGRU


class TestPo2Api(unittest.TestCase):
    def test_constructor_and_legacy_aliases(self):
        self.assertFalse(QuantGRU(3, 4).use_po2_scale)
        for model in (
            QuantGRU(3, 4, use_po2_scale=True),
            QuantGRU(3, 4, use_pot2_scale=True),
            QuantGRU(3, 4, 1, True, False, 0.0, False, False, True),
            QuantGRU(3, 4, use_po2_scale=True, use_pot2_scale=True),
        ):
            self.assertTrue(model.use_po2_scale)
            self.assertTrue(model.use_pot2_scale)
            model.use_pot2_scale = False
            self.assertFalse(model.use_po2_scale)
            model.use_po2_scale = True
            self.assertTrue(model.use_pot2_scale)
            self.assertTrue(copy.deepcopy(model).use_po2_scale)

    def test_constructor_rejects_conflicting_aliases(self):
        for canonical, legacy in ((True, False), (False, True)):
            with self.subTest(canonical=canonical, legacy=legacy):
                with self.assertRaisesRegex(ValueError, "use_po2_scale and use_pot2_scale conflict"):
                    QuantGRU(3, 4, use_po2_scale=canonical, use_pot2_scale=legacy)

    def test_config_names_and_legacy_checkpoint(self):
        for field in ("use_po2_scale", "use_pot2_scale"):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({"GRU_config": {"default_config": {field: True}}}))
                model = QuantGRU(3, 4)
                model.load_bitwidth_config(str(path))
                self.assertTrue(model.use_po2_scale)
                state = model.__getstate__()
                self.assertTrue(state["_bitwidth_config"]["usePOT2_"])
                restored = QuantGRU(3, 4)
                restored.__setstate__(state)
                self.assertTrue(restored.use_po2_scale)

    def test_config_rejects_conflicting_aliases_without_changing_mode(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"GRU_config": {"default_config": {
                "disable_quantization": False,
                "use_po2_scale": False,
                "use_pot2_scale": True,
            }}}))
            model = QuantGRU(3, 4)
            with self.assertRaisesRegex(ValueError, "use_po2_scale and use_pot2_scale conflict"):
                model.load_bitwidth_config(str(path))
            self.assertFalse(model.use_po2_scale)
            self.assertFalse(model.use_quantization)

    def test_methods_preserve_lock_behavior(self):
        for method in ("enable_po2", "enable_pot2"):
            with self.subTest(method=method):
                model = QuantGRU(3, 4)
                model.set_quant_params_locked(True)
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    getattr(model, method)()
                model.set_quant_params_locked(False)
                getattr(model, method)()
                self.assertTrue(model.use_po2_scale)
                model = QuantGRU(3, 4, use_po2_scale=True)
                model.set_quant_params_locked(True)
                getattr(model, method)()


if __name__ == "__main__":
    unittest.main()
