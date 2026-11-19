"""Compile-time activation selection applies to both independent GRU carriers."""

import tempfile
import unittest

import torch
import quant_gru


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class GruActivationModeTest(unittest.TestCase):
    def gate_case(self, bits, pot2, clipped_outputs=False):
        ops = quant_gru.gru_ops
        config = ops.OperatorQuantConfig()
        fields = ('x', 'h', 'W', 'R', 'bw', 'br', 'weight_ih_linear',
                  'weight_hh_linear', 'update_gate_input', 'update_gate_output',
                  'reset_gate_input', 'reset_gate_output', 'new_gate_input',
                  'new_gate_output', 'mul_reset_hidden', 'mul_old_contribution',
                  'mul_new_contribution')
        for field in fields:
            setattr(config, field + '_', bits)
            setattr(config, field + '_unsigned_', field in (
                'update_gate_output', 'reset_gate_output'))
            setattr(config, field + '_symmetric_', field in ('W', 'R', 'bw', 'br'))
        config.usePOT2_ = pot2
        parameters = ops.GRUQuantParams()
        parameters.hidden_ = 1
        parameters.bitwidth_config_ = config
        input_scale = (1.0 if pot2 else .75) / (1 << (bits - 4))
        for field in ('x', 'h', 'weight_ih_linear', 'weight_hh_linear',
                      'update_gate_input', 'reset_gate_input', 'new_gate_input',
                      'update_gate_output', 'reset_gate_output', 'new_gate_output',
                      'mul_reset_hidden', 'mul_old_contribution', 'mul_new_contribution'):
            setattr(parameters, 'scale_' + field + '_', input_scale)
            setattr(parameters, 'zp_' + field + '_', 0)
        for field in ('W', 'R', 'bw', 'br'):
            setattr(parameters, 'scale_' + field + '_', [1.0 / 32] * 3)
        parameters.zp_x_ = -3
        parameters.zp_h_ = 2
        for gate, zero_point in (('update', 5), ('reset', -7), ('new', 9)):
            setattr(parameters, 'zp_' + gate + '_gate_input_', zero_point)
        for gate in ('update', 'reset', 'new'):
            exponent = bits + 1 if clipped_outputs else bits - (2 if gate == 'new' else 1)
            setattr(parameters, 'scale_' + gate + '_gate_output_', 1.0 / (1 << exponent))
            setattr(parameters, 'zp_' + gate + '_gate_output_', 3 if gate == 'new' else 7)
        extent = (1 << (bits - 1)) * input_scale
        value = torch.linspace(-extent * 1.2, extent * 1.2,
                               4097 if bits == 16 else 513, device='cuda').reshape(1, -1, 1)
        arguments = dict(is_training=True, time_steps=1, batch_size=value.shape[1],
                         input_size=1, hidden_size=1, W=torch.ones(1, 3, device='cuda'),
                         R=torch.zeros(1, 3, device='cuda'), bw=torch.zeros(3, device='cuda'),
                         br=torch.zeros(3, device='cuda'), x=value,
                         h0=torch.zeros(value.shape[1], 1, device='cuda'), quant_params=parameters)
        return arguments, parameters

    def check_gate_values(self, carrier, bits, pot2, clipped_outputs):
        ops = quant_gru.gru_ops
        arguments, parameters = self.gate_case(bits, pot2, clipped_outputs)
        forward = getattr(ops, 'forward_quant_' + ('float' if carrier == 'float32' else 'int') + '_storage')
        result = forward(**arguments)
        cache = result[1][0]
        lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        # With W=1, R=bias=0 and matching linear/input grids, every preactivation
        # is the quantized input, followed only by linear and gate saturation.
        linear = (result[6][0, :, 0].float() - parameters.zp_x_).clamp(lower, upper)
        scale_x = ops.decode_effective_scale(parameters.scale_x_, pot2)
        for index, gate in enumerate(('update', 'reset', 'new')):
            zp_x = getattr(parameters, 'zp_' + gate + '_gate_input_')
            preactivation = linear + zp_x
            expected_input_mask = ((preactivation >= lower) & (preactivation <= upper)).to(torch.uint8)
            torch.testing.assert_close(result[15][0, :, index], expected_input_mask, rtol=0, atol=0)
            real_input = (preactivation.clamp(lower, upper) - zp_x) * scale_x
            activated = torch.tanh(real_input) if gate == 'new' else torch.sigmoid(real_input)
            scale_y = ops.decode_effective_scale(
                getattr(parameters, 'scale_' + gate + '_gate_output_'), pot2)
            zp_y = getattr(parameters, 'zp_' + gate + '_gate_output_')
            output_lower, output_upper = (lower, upper) if gate == 'new' else (0, (1 << bits) - 1)
            real_codes = torch.round(activated / scale_y) + zp_y
            expected_codes = real_codes.clamp(output_lower, output_upper)
            actual_codes = torch.round(cache[:, index] / scale_y) + zp_y
            self.assertTrue(torch.isfinite(cache[:, index]).all())
            self.assertTrue(((actual_codes >= output_lower) & (actual_codes <= output_upper)).all())
            if ops.use_real_activation:
                # A one-code allowance accounts for CUDA transcendental rounding.
                torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=1)
                expected_output_mask = ((real_codes >= output_lower) &
                                        (real_codes <= output_upper)).to(torch.uint8)
                torch.testing.assert_close(result[16][0, :, index], expected_output_mask, rtol=0, atol=0)
            else:
                error = (actual_codes - expected_codes).abs() * scale_y
                self.assertLessEqual(error.max().item(), .08 + 2 * scale_y)
                if bits == 16 and not clipped_outputs:
                    self.assertTrue(torch.any((actual_codes - expected_codes).abs() > 2),
                                    'LUT build unexpectedly matches the standard activation over the sweep')
        with torch.no_grad():
            arguments['is_training'] = False
            evaluated = forward(**arguments)
        torch.testing.assert_close(result[0], evaluated[0], rtol=0, atol=0)

    def test_three_gates_follow_the_compiled_activation_mode(self):
        self.assertIsInstance(quant_gru.gru_ops.use_real_activation, bool)
        for carrier in ('float32', 'int32'):
            for bits in (8, 16):
                for pot2 in (False, True):
                    for clipped in (False, True):
                        with self.subTest(carrier=carrier, bits=bits, pot2=pot2, clipped=clipped):
                            self.check_gate_values(carrier, bits, pot2, clipped)

    def test_carriers_execute_distinct_cuda_kernels(self):
        from torch.profiler import ProfilerActivity, profile, supported_activities

        if ProfilerActivity.CUDA not in supported_activities():
            self.skipTest('CUDA profiling is unavailable')
        arguments, _ = self.gate_case(8, False)
        for carrier, expected, excluded in (
            ('float', 'PointwiseOperationsFP', 'PointwiseOperationsQuant'),
            ('int', 'PointwiseOperationsQuant', 'PointwiseOperationsFP'),
        ):
            with self.subTest(carrier=carrier):
                forward = getattr(quant_gru.gru_ops, 'forward_quant_' + carrier + '_storage')
                forward(**arguments)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                    forward(**arguments)
                    torch.cuda.synchronize()
                names = [event.name for event in trace.events()]
                if not any('PointwiseOperations' in name for name in names):
                    self.skipTest('CUDA kernel events are unavailable in this profiler environment')
                self.assertTrue(any(expected in name for name in names))
                self.assertFalse(any(excluded in name for name in names))

    def test_each_carrier_backward_matches_the_ste_reference(self):
        from tests.recurrent.test_gru_autograd import surrogate, swap_gates
        from tests.support.recurrent import fixed_gru_model

        for carrier in ('float32', 'int32'):
            for clipped in (None, 'weight_ih_linear', 'new_gate_input', 'new_gate_output'):
                with self.subTest(carrier=carrier, clipped=clipped), tempfile.TemporaryDirectory() as directory:
                    model, encodings = fixed_gru_model(directory, clipped)
                    model.quant_storage_dtype = carrier
                    value = torch.linspace(.35, .65, 6, device='cuda').reshape(3, 2, 1).requires_grad_()
                    initial = torch.full((1, 2, 1), .2, device='cuda', requires_grad=True)
                    output, hidden = model(value, initial)
                    saved = tuple(tensor.detach().clone() for tensor in output.grad_fn.saved_tensors)
                    expected, expected_hidden, leaves = surrogate(saved, encodings)
                    loss = output.square().sum() + hidden.sum()
                    expected_loss = expected.square().sum() + expected_hidden.sum()
                    loss.backward()
                    expected_loss.backward()
                    actual = (swap_gates(model.weight_ih_l0.grad).t(),
                              swap_gates(model.weight_hh_l0.grad).t(),
                              swap_gates(model.bias_ih_l0.grad),
                              swap_gates(model.bias_hh_l0.grad), value.grad, initial.grad.squeeze(0))
                    for gradient, leaf in zip(actual, leaves):
                        torch.testing.assert_close(gradient, leaf.grad, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
