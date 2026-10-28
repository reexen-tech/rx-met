"""Equivalent KWS graph coverage; this does not execute the ADA200 compiler."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gru_interface_binding
import numpy as np
import onnx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aimet_common import run_context as runtime
from aimet_common.compiler_encodings import validate_compiler_encodings
from aimet_torch.rx_export.export_onnx_json import export_onnx_json
from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth, apply_power_of_2_workflow
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from examples import quick_start_kws as kws
from tests.support.kws_artifacts import validate_ada200_artifacts
from tests.support.kws_reference_model import (
    ReferenceAttMHRNN,
    ReferenceConvActBN,
    save_reference_checkpoint,
)
from tests.support.encodings import (
    assert_public_encoding_document,
    assert_recurrent_encoding_schema,
)


MODEL_OPTIONS = dict(cnn_filters=(2, 1), rnn_layers=2, rnn_units=4,
                     heads=4, dense_units=(8, 4), dropout=0.)


class KwsCompilerGraphTest(unittest.TestCase):
    def setUp(self):
        original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', original_matmul_tf32)
        self.addCleanup(setattr, torch.backends.cudnn, 'allow_tf32', original_cudnn_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        runtime.configure_output(timestamp_output=False)
        self.addCleanup(runtime.configure_output)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(runtime.execution(self.directory))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        # CPU native recurrent wheels implement inference, without autograd.
        stack.enter_context(torch.no_grad())
        if gru_interface_binding.cpu_only:
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.skipTest('The CUDA recurrent wheel requires a CUDA device')
        torch.manual_seed(53)

    def model(self, rnn_type='gru'):
        return kws.AttMHRNN(rnn_type=rnn_type, **MODEL_OPTIONS).to(self.device).eval()

    def sim(self, *, calibrate):
        model = kws.prepare_model(self.model()).to(self.device).eval()
        inputs = torch.linspace(-2., 2., 49 * 20, device=self.device).reshape(1, 49, 20)
        sim = QuantizationSimModel(model, inputs, quant_scheme='tf',
                                   config_file=str(kws.CONFIG_FILE),
                                   default_output_bw=8, default_param_bw=8)
        apply_mixed_precision_bitwidth(sim.model, str(kws.BITWIDTH_CONFIG_FILE), verbose=False)
        kws.configure_attention_quantizers(sim.model)
        if calibrate:
            with compute_encodings(sim.model):
                sim.model(inputs)
                sim.model(inputs.flip(1) * .7)
        return sim, inputs

    def test_cli_keeps_original_outputs_and_accepts_legacy_checkpoint(self):
        args = kws.parse_args(['--device', 'cpu', '--ptq-only',
                               '--fp-checkpoint', '/models/original.pth'])
        self.assertTrue(args.ptq_only)
        self.assertEqual(args.fp_checkpoint, Path('/models/original.pth'))
        self.assertEqual(args.quant_config, kws.BITWIDTH_CONFIG_FILE)
        self.assertTrue(args.quant_config.is_file())
        self.assertEqual(args.output_dir, kws.OUTPUT_DIR)
        self.assertEqual(kws.BITWIDTH_CONFIG_FILE.name, 'quick_start_full_quant.json')
        with self.assertRaises(SystemExit):
            kws.parse_args(['--device', 'cpu'])

    def test_original_checkpoints_preserve_all_weights_and_float_outputs(self):
        waveform = torch.randn(2, kws.TARGET_LEN, device=self.device) * .02
        for rnn_type in ('gru', 'lstm'):
            with self.subTest(rnn_type=rnn_type):
                baseline = ReferenceAttMHRNN(rnn_type=rnn_type, **MODEL_OPTIONS)
                baseline = baseline.to(self.device).eval()
                checkpoint = self.directory / f'{rnn_type}-original.pth'
                save_reference_checkpoint(baseline, checkpoint, rnn_type)
                adapted = self.model(rnn_type)
                frontend = kws.make_speech_features(device=self.device).eval()
                kws.load_checkpoint(adapted, checkpoint, rnn_type, self.device,
                                    speech_features=frontend)
                torch.testing.assert_close(frontend(waveform), baseline.speech_features(waveform),
                                           rtol=0, atol=0)
                features = frontend(waveform)
                self.assertEqual(tuple(features.shape), (2, 49, 20))
                self.assertFalse(any(isinstance(module, nn.Linear) for module in adapted.modules()))
                self.assertNotIn('speech_features', dict(adapted.named_modules()))
                linear_count = 0
                for name, module in baseline.named_modules():
                    if not isinstance(module, nn.Linear):
                        continue
                    linear_count += 1
                    convolution = adapted.get_submodule(name).conv
                    self.assertEqual(convolution.kernel_size, (1, 1))
                    torch.testing.assert_close(convolution.weight[..., 0, 0], module.weight,
                                               rtol=0, atol=0)
                    torch.testing.assert_close(convolution.bias, module.bias, rtol=0, atol=0)
                self.assertEqual(linear_count, 7)
                torch.testing.assert_close(adapted(features), baseline(waveform),
                                           atol=2e-6, rtol=2e-5)

    def test_mfcc_feeds_are_computed_outside_the_exported_network(self):
        waveform = np.linspace(-.02, .02, kws.TARGET_LEN, dtype=np.float32).reshape(1, -1)
        feeds = [{'input': waveform}, {'input': -waveform}]
        frontend = kws.make_speech_features(device=self.device).eval()
        converted = kws.extract_mfcc_feeds(feeds, frontend, self.device)
        self.assertEqual(len(converted), len(feeds))
        for source, result in zip(feeds, converted):
            self.assertEqual(set(result), {'input'})
            self.assertEqual(result['input'].shape, (1, 49, 20))
            self.assertEqual(result['input'].dtype, np.dtype('float32'))
            expected = frontend(torch.from_numpy(source['input']).to(self.device)).cpu().numpy()
            np.testing.assert_array_equal(result['input'], expected)
        self.assertEqual(feeds[0]['input'].shape, (1, kws.TARGET_LEN))

    def test_single_channel_bn_layout_preserves_training_and_inference(self):
        reference = ReferenceConvActBN(3, 1, (5, 1)).to(self.device).train()
        adapted = kws.ConvActBN(3, 1, (5, 1)).to(self.device).train()
        adapted.load_state_dict(reference.state_dict(), strict=True)
        with torch.enable_grad():
            for _ in range(2):
                reference.zero_grad(set_to_none=True)
                adapted.zero_grad(set_to_none=True)
                source = torch.randn(2, 3, 49, 20, device=self.device).requires_grad_()
                candidate = source.detach().clone().requires_grad_()
                expected = reference(source)
                actual = adapted(candidate)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                actual.backward(gradient)
                torch.testing.assert_close(candidate.grad, source.grad, atol=1e-6, rtol=1e-5)
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in adapted.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad, reference_parameters[name].grad,
                                               atol=1e-6, rtol=1e-5)
                for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                    torch.testing.assert_close(getattr(adapted.bn, name), getattr(reference.bn, name),
                                               atol=1e-6, rtol=1e-5)
        self.assertEqual(adapted.bn.num_batches_tracked.item(), 2)
        reference.eval()
        adapted.eval()
        inputs = torch.randn(2, 3, 49, 20, device=self.device)
        torch.testing.assert_close(adapted(inputs), reference(inputs), atol=1e-6, rtol=1e-5)

    def test_checkpoint_conversion_rejects_missing_core_parameters(self):
        baseline = ReferenceAttMHRNN(**MODEL_OPTIONS).to(self.device).eval()
        state = baseline.state_dict()
        del state['query_projs.0.weight']
        checkpoint = self.directory / 'incomplete.pth'
        torch.save({'rnn_type': 'gru', 'model': state}, checkpoint)
        with self.assertRaises((RuntimeError, ValueError)):
            kws.load_checkpoint(self.model(), checkpoint, 'gru', self.device,
                                speech_features=kws.make_speech_features(device=self.device))

    def test_cuda_default_training_workflow_exports_and_restores(self):
        if self.device.type != 'cuda':
            self.skipTest('Floating-point training and QAT require the CUDA recurrent wheel')
        waveforms = torch.randn(2, kws.TARGET_LEN) * .02
        data = TensorDataset(waveforms, torch.tensor([2, 3]))
        loader = DataLoader(data, batch_size=2, shuffle=False, num_workers=0)
        loaders = {name: loader for name in ('train', 'val', 'test', 'calib')}
        output_root = self.directory / 'training'
        original_device = kws.DEVICE
        self.addCleanup(setattr, kws, 'DEVICE', original_device)
        with torch.enable_grad(), patch.dict(os.environ), \
             patch.object(kws, 'build_dataloaders', return_value=loaders), \
             patch.object(kws, '_new_model', side_effect=self.model), \
             patch.object(kws, 'train_floating_point', wraps=kws.train_floating_point) as train, \
             patch.object(kws, 'qat_finetune', wraps=kws.qat_finetune) as qat:
            os.environ.pop('RX_MET_KWS_FP_MODEL', None)
            kws.main(['--rnn_type', 'gru', '--output-dir', str(output_root)])
        train.assert_called_once()
        qat.assert_called_once()
        execution = output_root / 'gru'
        self.assertEqual(json.loads((execution / 'run.json').read_text())['status'], 'success')
        self.assertTrue((execution / 'run.log').is_file())
        for filename in ('model_fp_kws.pth', 'att_mh_rnn_kws_qat.pth'):
            checkpoint_path = execution / 'checkpoints' / filename
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['rnn_type'], 'gru')
            self.assertEqual(checkpoint['model_variant'], 'ada200-equivalent-v1')
            self.assertTrue(checkpoint['speech_features'])
            self.assertFalse(any(name.startswith('speech_features.') for name in checkpoint['model']))
        export_root = execution / 'exports'
        validate_ada200_artifacts(export_root / 'att_mh_rnn_kws.onnx',
                                  export_root / 'att_mh_rnn_kws.json')

    def test_attention_rewrite_preserves_outputs_and_gradients(self):
        with torch.enable_grad():
            for scale in (.05, 1., 5.):
                sequence = (torch.randn(2, 49, 8, device=self.device) * scale).requires_grad_()
                query = torch.randn(2, 1, 8, device=self.device, requires_grad=True)
                candidate_sequence = sequence.detach().clone().requires_grad_()
                candidate_query = query.detach().clone().requires_grad_()
                logits = torch.matmul(sequence, query.transpose(1, 2))
                expected = torch.matmul(torch.softmax(logits, dim=1).transpose(1, 2), sequence)
                actual = kws.AttentionHead(49)(candidate_sequence, candidate_query)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                actual.backward(gradient)
                torch.testing.assert_close(candidate_sequence.grad, sequence.grad,
                                           atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(candidate_query.grad, query.grad,
                                           atol=2e-5, rtol=2e-4)

    def test_four_head_graph_calibrates_exports_and_restores(self):
        sim, inputs = self.sim(calibrate=True)
        report = apply_power_of_2_workflow(sim.model, align_bias_scale=True, verbose=False)
        self.assertTrue(report['all_power_of_2'])
        self.assertEqual(report['verification']['native']['module_count'], 2)
        expected = sim.model(inputs).detach().clone()
        self.assertTrue(torch.isfinite(expected).all())
        onnx_path, json_path = export_onnx_json(
            sim, str(self.directory), 'ada200-four-head', (1, 49, 20), opset=18)
        graph = onnx.load(onnx_path)
        onnx.checker.check_model(graph)
        self.assertEqual([dimension.dim_value for dimension in graph.graph.input[0].type.tensor_type.shape.dim],
                         [1, 49, 20])
        operators = [node.op_type for node in graph.graph.node]
        self.assertFalse({'Log', 'Softmax', 'Gemm', 'MatMul'}.intersection(operators))
        self.assertEqual(operators.count('GRU'), 2)
        self.assertEqual(operators.count('Conv'), 9)
        self.assertEqual(operators.count('Exp'), 4)
        self.assertEqual(operators.count('ReduceMax'), 4)
        self.assertGreaterEqual(operators.count('ReduceSum'), 12)
        self.assertEqual(sum(node.op_type == 'Conv' and 'query_projs.' in node.name
                             for node in graph.graph.node), 4)
        document = json.loads(Path(json_path).read_text())
        assert_public_encoding_document(self, document)
        validate_compiler_encodings(document, strict=True, require_po2=True)
        initializers = {value.name for value in graph.graph.initializer}
        for node in graph.graph.node:
            if node.op_type == 'GRU':
                entry = document['activation_encodings'][node.name]
                self.assertTrue(entry['is_GRU'])
                assert_recurrent_encoding_schema(self, entry)
            if node.op_type == 'Conv':
                self.assertIn(node.name, document['activation_encodings'])
                self.assertIn(node.input[1], initializers)
                self.assertIn(node.input[1], document['param_encodings'])
                self.assertIn(node.input[2], document['param_encodings'])
                self.assertIn('scale', document['param_encodings'][node.input[1]])
                self.assertIn('n', document['param_encodings'][node.input[1]])
        validate_ada200_artifacts(onnx_path, json_path)
        unsupported_graph = onnx.load(onnx_path)
        next(node for node in unsupported_graph.graph.node if node.op_type == 'Exp').op_type = 'Softmax'
        unsupported_path = self.directory / 'unsupported.onnx'
        onnx.save(unsupported_graph, unsupported_path)
        with self.assertRaises(ValueError):
            validate_ada200_artifacts(unsupported_path, json_path)
        projection = next(node for node in graph.graph.node
                          if node.op_type == 'Conv' and 'query_projs.' in node.name)
        del document['param_encodings'][projection.input[1]]
        incomplete_path = self.directory / 'missing-projection.json'
        incomplete_path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            validate_ada200_artifacts(onnx_path, incomplete_path)
        restored, _ = self.sim(calibrate=False)
        quantizer_keys = {f'{name}.{key}' for name, module in sim.model.named_modules(remove_duplicate=False)
                          if isinstance(module, QuantizerBase) for key in module.state_dict()}
        state = {name: value for name, value in sim.model.state_dict().items()
                 if name not in quantizer_keys}
        incompatible = restored.model.load_state_dict(state, strict=False)
        restored_quantizer_keys = {
            f'{name}.{key}' for name, module in restored.model.named_modules(remove_duplicate=False)
            if isinstance(module, QuantizerBase) for key in module.state_dict()
        }
        self.assertEqual(set(incompatible.missing_keys), restored_quantizer_keys)
        self.assertEqual(incompatible.unexpected_keys, [])
        loaded = load_quantizer_encodings(restored.model, str(json_path),
                                          allow_overwrite=False, verbose=False)
        self.assertGreater(loaded['loaded'], 2)
        torch.testing.assert_close(restored.model(inputs), expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
