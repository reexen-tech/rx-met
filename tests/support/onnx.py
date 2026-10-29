"""Small real ONNX models and PTQ fixtures shared across test domains."""

import json
from pathlib import Path
import tempfile

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow, disable_onnx_bias_quantizers


def make_graph(*, bias=True, second_layer=False, matmul_add=False):
    weights = np.array([[.01, -.02, .01, -.02], [-.02, .01, -.02, .01]], dtype=np.float32)
    initializers = [numpy_helper.from_array(weights.T if matmul_add or not bias else weights, 'w')]
    nodes = []
    output = 'mid' if second_layer else 'y'
    if bias:
        initializers.append(numpy_helper.from_array(np.array([.5, -.25], dtype=np.float32), 'b'))
    if bias and not matmul_add:
        nodes.append(helper.make_node('Gemm', ['x', 'w', 'b'], [output], name='fc', transB=1))
    else:
        nodes.append(helper.make_node('MatMul', ['x', 'w'], ['product' if bias else output], name='fc'))
        if bias:
            nodes.append(helper.make_node('Add', ['product', 'b'], [output], name='add'))
    if second_layer:
        initializers.extend([
            numpy_helper.from_array(np.array([[.1, -.2], [-.2, .1]], dtype=np.float32), 'w2'),
            numpy_helper.from_array(np.array([.1, -.1], dtype=np.float32), 'b2'),
        ])
        nodes.append(helper.make_node('Gemm', ['mid', 'w2', 'b2'], ['y'], name='fc2', transB=1))
    graph = helper.make_graph(nodes, 'alignment',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])], initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)


class OnnxFixture:
    def __init__(self, testcase):
        temporary = tempfile.TemporaryDirectory(prefix='onnx-ptq-')
        testcase.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.config = self.path / 'config.json'
        self.config.write_text(json.dumps({
            'defaults': {'ops': {'is_output_quantized': 'True', 'is_symmetric': 'True'},
                         'params': {'is_quantized': 'True', 'is_symmetric': 'True'},
                         'per_channel_quantization': 'True', 'strict_symmetric': 'False',
                         'unsigned_symmetric': 'False'},
            'params': {}, 'op_type': {}, 'supergroups': [], 'model_input': {}, 'model_output': {},
        }))
        self.feed = {'x': np.array([[.1, -.1, .1, -.1]], dtype=np.float32)}
        self.index = 0

    def sim(self, **kwargs):
        self.index += 1
        model = make_graph(**kwargs)
        sim = pipeline.create_onnx_ptq_sim(model, self.feed, config_file=str(self.config),
            quant_scheme='tf', work_dir=self.path / f'sim{self.index}')
        return sim, model

    def calibrate(self, sim, *, bias=True):
        if bias:
            disable_onnx_bias_quantizers(sim)
        sim.compute_encodings([self.feed, {'x': self.feed['x'] + 1.0}])

    def apply_config(self, sim, config):
        path = self.path / 'mixed.json'
        path.write_text(json.dumps(config))
        apply_onnx_mixed_precision_bitwidth(sim, str(path), verbose=False)

    def artifacts(self, *, matmul_add=False, mixed=False):
        sim, model = self.sim(matmul_add=matmul_add)
        if mixed:
            self.apply_config(sim, {'layer_type_config': {'QuantizedLinear': {
                'weight_bitwidth': 16, 'input_bitwidth': 8, 'output_bitwidth': 16,
            }}})
        self.calibrate(sim)
        apply_power_of_2_workflow(sim, verbose=False)
        expected = pipeline.run_onnx_feed(sim, self.feed)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / f'export{self.index}', 'model')
        return artifacts, expected

    def verify(self, artifacts, expected):
        return pipeline.verify_onnx_ptq_artifacts(artifacts, self.feed, expected,
            config_file=str(self.config), quant_scheme='tf')


class OnnxGraphFixture(OnnxFixture):
    def model(self, nodes, input_shapes, output_shapes, constants=None, *, quant_scheme='tf'):
        def infos(shapes):
            return [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
                    for name, shape in shapes.items()]
        graph = helper.make_graph(nodes, 'operand-precision', infos(input_shapes), infos(output_shapes),
                                  [numpy_helper.from_array(value, name)
                                   for name, value in (constants or {}).items()])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        feeds = {name: np.linspace(-1, 1, np.prod(shape), dtype=np.float32).reshape(shape)
                 for name, shape in input_shapes.items()}
        self.index += 1
        sim = pipeline.create_onnx_ptq_sim(model, feeds, config_file=str(self.config),
            quant_scheme=quant_scheme, work_dir=self.path / f'graph{self.index}')
        return sim, model, feeds

    def apply(self, sim, config):
        self.apply_config(sim, config)

    def moved_sim(self):
        return self.model([
            helper.make_node('Transpose', ['x'], ['moved'], name='transpose', perm=[1, 0]),
            helper.make_node('Mul', ['moved', 'gain'], ['y'], name='window'),
        ], {'x': [2, 4]}, {'y': [4, 2]}, {'gain': np.ones((4, 2), dtype=np.float32)})
