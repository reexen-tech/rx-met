"""Omitted symmetry defaults to symmetric without overriding explicit choices."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from aimet_common import libquant_info
from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.json_config_importer import JsonConfigImporter
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode
from aimet_onnx.rx_ptq.pipeline import create_onnx_ptq_sim
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.quantization.affine import Quantize, QuantizeDequantize
from tests.support import onnx as fixtures


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


def config():
    return {'defaults': {'ops': {'is_output_quantized': 'True'},
                         'params': {'is_quantized': 'True'}},
            'params': {}, 'op_type': {}, 'supergroups': [],
            'model_input': {}, 'model_output': {}}


class SymmetricDefaultsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='symmetric-defaults-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'config.json'

    def test_primitive_defaults_are_symmetric_in_both_integer_domains(self):
        for signed in (True, False):
            for cls in (Quantize, QuantizeDequantize):
                q = cls((), bitwidth=8, signed=signed)
                self.assertTrue(q.symmetric)
                self.assertEqual(q.signed, signed)
            native = QcQuantizeOp(libquant_info.QcQuantizeInfo(),
                                  op_mode=OpMode.updateStats, signed=signed)
            self.assertTrue(native.use_symmetric_encodings)
            self.assertEqual(native.signed, signed)

    def test_shared_config_import_fills_only_missing_symmetry(self):
        for explicit in (None, True, False):
            source = config()
            if explicit is not None:
                source['defaults']['ops']['is_symmetric'] = str(explicit)
            self.path.write_text(json.dumps(source))
            parsed = JsonConfigImporter.import_json_config_file(str(self.path))
            self.assertEqual(parsed['defaults']['ops']['is_symmetric'],
                             True if explicit is None else explicit)
            self.assertTrue(parsed['defaults']['params']['is_symmetric'])

    def test_torch_and_onnx_config_omissions_use_symmetric_grids(self):
        values = torch.tensor([[-1., .25, 1., 2.]])
        for explicit in (None, False):
            with self.subTest(explicit=explicit):
                source = config()
                if explicit is not None:
                    for section in ('ops', 'params'):
                        source['defaults'][section]['is_symmetric'] = str(explicit)
                self.path.write_text(json.dumps(source))
                sim = QuantizationSimModel(prepare_model(Model().eval()), values,
                    quant_scheme='tf', config_file=str(self.path))
                sim.compute_encodings(lambda model: model(values))
                expected = explicit is None
                for module in sim.model.modules():
                    if isinstance(module, QuantizationMixin):
                        for q in list(module.input_quantizers) + list(module.output_quantizers) + list(module.param_quantizers.values()):
                            if q is not None:
                                self.assertEqual(q.symmetric, expected)
                native = create_onnx_ptq_sim(fixtures.make_graph(), {'x': values.numpy()},
                    quant_scheme='tf', config_file=str(self.path), work_dir=self.root / f'onnx-{explicit}')
                native.compute_encodings([{'x': values.numpy()}])
                for q in native.qc_quantize_op_dict.values():
                    if q.enabled:
                        self.assertEqual(q.use_symmetric_encodings, expected)
                        if expected:
                            points = [-(2 ** (e.bw-1))-e.offset for e in q.get_encodings()]
                            self.assertTrue(all(point == 0 for point in points))

    def test_lazy_wrapper_defaults_are_symmetric(self):
        from aimet_torch.v2._builder import _V2LazyQuantizeWrapper
        module = QuantizationMixin.from_module(torch.nn.Linear(4, 2))
        wrapper = _V2LazyQuantizeWrapper(module, 8, 8, 'nearest', QuantScheme.post_training_tf)
        for q in list(wrapper.input_quantizers) + list(wrapper.output_quantizers) + list(wrapper.param_quantizers.values()):
            self.assertTrue(q.use_symmetric_encodings)


if __name__ == '__main__':
    unittest.main()
