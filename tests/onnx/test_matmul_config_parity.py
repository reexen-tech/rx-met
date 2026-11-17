"""MatMul input overrides retain their meaning across ONNX graph layouts."""

import json
import unittest

import numpy as np
from onnx import helper

from aimet_onnx.rx_ptq.input_quantization import input_quantizer
from tests.support import onnx as fixtures


class MatMulConfigurationParityTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxGraphFixture(self)

    def test_input_list_overrides_global_activation_precision(self):
        for split_linear in (False, True):
            with self.subTest(split_linear=split_linear):
                sim, _ = self.fixture.sim(bias=split_linear, matmul_add=split_linear)
                self.fixture.apply_config(sim, {
                    'default_bitwidth': {'weight': 16, 'output': 16},
                    'layer_name_config': {'fc': {'input_bitwidths': [8, 8]}},
                })
                op = sim.connected_graph.get_all_ops()['fc']
                self.assertEqual(input_quantizer(sim, op, 0).bitwidth, 8)
                self.assertEqual(sim.qc_quantize_op_dict['w'].bitwidth, 8)

    def test_matmul_input_override_keeps_producer_output_precision(self):
        sim, _, _ = self.fixture.model([
            helper.make_node('Relu', ['x'], ['before'], name='producer'),
            helper.make_node('MatMul', ['before', 'w'], ['y'], name='matmul'),
        ], {'x': [1, 4]}, {'y': [1, 2]},
            {'w': np.ones((4, 2), dtype=np.float32)})
        self.fixture.apply_config(sim, {
            'default_bitwidth': {'weight': 16, 'output': 16},
            'layer_name_config': {
                'producer': {'output_bitwidth': 16},
                'matmul': {'input_bitwidths': [8, 8]},
            },
        })
        op = sim.connected_graph.get_all_ops()['matmul']
        self.assertEqual(sim.qc_quantize_op_dict['before'].bitwidth, 16)
        self.assertEqual(input_quantizer(sim, op, 0).bitwidth, 8)
        self.assertEqual(input_quantizer(sim, op, 1).bitwidth, 8)

    def test_dynamic_matmul_input_settings_match_torch(self):
        import torch
        from torch import nn
        from aimet_torch._base.nn.modules.custom import MatMul
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        from aimet_torch.v2.quantsim import QuantizationSimModel

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.producer = nn.ReLU()
                self.matmul = MatMul()

            def forward(self, x, right):
                return self.matmul(self.producer(x), right)

        for widths, expected in (([8, 8], [8, 8]), ([8, None], [8, 16])):
            with self.subTest(input_bitwidths=widths):
                config = {
                    'default_bitwidth': {'weight': 16, 'output': 16},
                    'layer_name_config': {
                        'producer': {'output_bitwidth': 16},
                        'matmul': {'input_bitwidth': 16, 'input_bitwidths': widths},
                    },
                }
                torch_sim = QuantizationSimModel(Model().eval(),
                    (torch.ones(1, 4), torch.ones(4, 2)),
                    config_file=str(self.fixture.config), default_output_bw=16)
                config_path = self.fixture.path / 'torch-mixed.json'
                config_path.write_text(json.dumps(config))
                apply_mixed_precision_bitwidth(torch_sim.model, str(config_path), verbose=False)
                torch_inputs = [q.bitwidth for q in torch_sim.model.matmul.input_quantizers]
                self.assertEqual(torch_inputs, expected)
                self.assertEqual(torch_sim.model.producer.output_quantizers[0].bitwidth, 16)

                onnx_sim, _, _ = self.fixture.model([
                    helper.make_node('Relu', ['x'], ['before'], name='producer'),
                    helper.make_node('MatMul', ['before', 'right'], ['y'], name='matmul'),
                ], {'x': [1, 4], 'right': [4, 2]}, {'y': [1, 2]})
                # Match the Torch construction width before testing null preservation.
                for quantizer in onnx_sim.qc_quantize_op_dict.values():
                    quantizer.set_bitwidth(16)
                self.fixture.apply_config(onnx_sim, config)
                op = onnx_sim.connected_graph.get_all_ops()['matmul']
                self.assertEqual(
                    [input_quantizer(onnx_sim, op, index).bitwidth for index in range(2)],
                    torch_inputs,
                )
                self.assertEqual(onnx_sim.qc_quantize_op_dict['before'].bitwidth,
                    torch_sim.model.producer.output_quantizers[0].bitwidth)

    def test_named_matmul_operand_settings_ignore_unrelated_linear_type_rule(self):
        for type_config in ({}, {'QuantizedLinear': {'output_bitwidth': 16}}):
            with self.subTest(layer_type_config=type_config):
                sim, _ = self.fixture.sim(bias=False)
                for quantizer in sim.qc_quantize_op_dict.values():
                    quantizer.set_bitwidth(16)
                self.fixture.apply_config(sim, {
                    'default_bitwidth': {'weight': 16, 'output': 16},
                    'layer_type_config': type_config,
                    'layer_name_config': {'fc': {'input_bitwidths': [8, 8]}},
                })
                op = sim.connected_graph.get_all_ops()['fc']
                self.assertEqual(input_quantizer(sim, op, 0).bitwidth, 8)
                self.assertEqual(input_quantizer(sim, op, 1).bitwidth, 8)

    def test_repeated_graph_input_has_independent_operand_grids(self):
        sim, _, _ = self.fixture.model([
            helper.make_node('MatMul', ['x', 'x'], ['y'], name='matmul'),
        ], {'x': [2, 2]}, {'y': [2, 2]})
        self.fixture.apply_config(sim, {
            'default_bitwidth': {'output': 16},
            'layer_name_config': {'matmul': {'input_bitwidths': [8, 16]}},
        })
        op = sim.connected_graph.get_all_ops()['matmul']
        left, right = [input_quantizer(sim, op, index) for index in range(2)]
        self.assertIsNot(left, right)
        self.assertEqual([left.bitwidth, right.bitwidth], [8, 16])
        self.assertFalse(sim.qc_quantize_op_dict['x'].enabled)

    def test_matmul_branches_keep_separate_input_precision(self):
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                nodes = [helper.make_node('Relu', ['x'], ['before'], name='producer')]
                left = 'before'
                if transpose:
                    nodes.append(helper.make_node('Transpose', ['before'], ['moved'],
                        name='transpose', perm=[1, 0]))
                    left = 'moved'
                nodes.extend([
                    helper.make_node('MatMul', [left, 'wa'], ['ya'], name='branch_a'),
                    helper.make_node('MatMul', ['before', 'wb'], ['yb'], name='branch_b'),
                ])
                sim, _, _ = self.fixture.model(nodes, {'x': [2, 2]},
                    {'ya': [2, 2], 'yb': [2, 2]},
                    {'wa': np.ones((2, 2), dtype=np.float32),
                     'wb': np.ones((2, 2), dtype=np.float32)})
                self.fixture.apply_config(sim, {
                    'default_bitwidth': {'weight': 16, 'output': 16},
                    'layer_name_config': {
                        'producer': {'output_bitwidth': 16},
                        'branch_a': {'input_bitwidths': [8, 8]},
                        'branch_b': {'input_bitwidths': [16, 16]},
                    },
                })
                ops = sim.connected_graph.get_all_ops()
                for name, width in (('branch_a', 8), ('branch_b', 16)):
                    self.assertEqual(
                        [input_quantizer(sim, ops[name], index).bitwidth for index in range(2)],
                        [width, width],
                    )
                self.assertIsNot(input_quantizer(sim, ops['branch_a'], 0),
                    input_quantizer(sim, ops['branch_b'], 0))
                self.assertTrue(sim.qc_quantize_op_dict['before'].enabled)
                self.assertEqual(sim.qc_quantize_op_dict['before'].bitwidth, 16)

    def test_normalized_matmul_keeps_named_operand_precedence(self):
        cases = (
            ({'input_bitwidth': 8}, 8),
            ({'input_bitwidth': 8, 'input_bitwidths': [8, None]}, 16),
            ({'input_bitwidth': 8, 'weight_bitwidth': 16}, 16),
            ({'input_bitwidths': [8, 8], 'weight_bitwidth': 16}, 16),
        )
        for settings, weight_width in cases:
            with self.subTest(settings=settings):
                sim, _ = self.fixture.sim(matmul_add=True)
                for quantizer in sim.qc_quantize_op_dict.values():
                    quantizer.set_bitwidth(16)
                self.fixture.apply_config(sim, {
                    'default_bitwidth': {'weight': 16, 'output': 16},
                    'layer_name_config': {'fc': settings},
                })
                op = sim.connected_graph.get_all_ops()['fc']
                self.assertEqual(input_quantizer(sim, op, 0).bitwidth, 8)
                self.assertEqual(sim.qc_quantize_op_dict['w'].bitwidth, weight_width)

    def test_normalized_matmul_keeps_named_operand_signedness(self):
        sim, _ = self.fixture.sim(matmul_add=True)
        self.fixture.apply_config(sim, {
            'layer_name_config': {'fc': {
                'input_bitwidths': [8, 8],
                'input_signeds': [True, False],
            }},
        })
        op = sim.connected_graph.get_all_ops()['fc']
        self.assertTrue(input_quantizer(sim, op, 0).signed)
        self.assertFalse(sim.qc_quantize_op_dict['w'].signed)


if __name__ == '__main__':
    unittest.main()
