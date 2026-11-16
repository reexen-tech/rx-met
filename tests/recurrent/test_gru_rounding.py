"""GRU carriers round centered values before translating by the zero point."""

import unittest

import torch
import quant_gru


@unittest.skipUnless(quant_gru._CPU_ONLY or torch.cuda.is_available(),
                     'CUDA is required for a CUDA QuantGRU build')
class GruRoundingTest(unittest.TestCase):
    def fixture(self, bits, pot2):
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
        for field in fields:
            setattr(parameters, 'scale_' + field + '_',
                    [1.0 / 32] * 3 if field in ('W', 'R', 'bw', 'br') else 1.0)
            if field not in ('W', 'R', 'bw', 'br'):
                setattr(parameters, 'zp_' + field + '_', 0)
        device = torch.device('cpu' if quant_gru._CPU_ONLY else 'cuda')
        arguments = dict(is_training=not quant_gru._CPU_ONLY,
                         time_steps=1, batch_size=1, input_size=1, hidden_size=1,
                         W=torch.zeros(1, 3, device=device),
                         R=torch.zeros(1, 3, device=device),
                         bw=torch.zeros(3, device=device), br=torch.zeros(3, device=device),
                         x=torch.zeros(1, 1, 1, device=device),
                         h0=torch.zeros(1, 1, device=device), quant_params=parameters)
        return arguments, parameters

    def forward(self, carrier, arguments):
        suffix = 'float' if carrier == 'float32' else 'int'
        operation = getattr(quant_gru.gru_ops, 'forward_quant_' + suffix + '_storage')
        return operation(**arguments)

    def test_input_and_initial_state_round_before_zero_point(self):
        for carrier in ('float32', 'int32'):
            for bits in (8, 16):
                for pot2 in (False, True):
                    for zero_point in (-3, 3):
                        with self.subTest(carrier=carrier, bits=bits,
                                          pot2=pot2, zero_point=zero_point):
                            arguments, parameters = self.fixture(bits, pot2)
                            device = arguments['x'].device
                            lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1

                            def centered_values(zp):
                                return torch.tensor([
                                    lower - zp - 1.5, lower - zp - .5, lower - zp + .5,
                                    -2.5, -1.5, -.5, .5, 1.5, 2.5,
                                    upper - zp - .5, upper - zp + .5, upper - zp + 1.5,
                                ], device=device)

                            x_centered = centered_values(zero_point)
                            h_centered = centered_values(-zero_point).flip(0)
                            scale = .5 if pot2 else .75
                            parameters.scale_x_ = scale
                            parameters.scale_h_ = scale
                            parameters.zp_x_ = zero_point
                            parameters.zp_h_ = -zero_point
                            arguments['batch_size'] = x_centered.numel()
                            arguments['x'] = (x_centered * scale).reshape(1, -1, 1)
                            arguments['h0'] = (h_centered * scale).reshape(-1, 1)
                            result = self.forward(carrier, arguments)

                            x_raw = (torch.round(x_centered) + zero_point).reshape(1, -1, 1)
                            h_raw = (torch.round(h_centered) - zero_point).reshape(-1, 1)
                            expected_x = x_raw.clamp(lower, upper)
                            expected_h = (h_raw.clamp(lower, upper) + zero_point) * scale
                            torch.testing.assert_close(result[6].float(), expected_x, rtol=0, atol=0)
                            torch.testing.assert_close(result[0][0], expected_h, rtol=0, atol=0)
                            if not quant_gru._CPU_ONLY:
                                expected_x_mask = ((x_raw >= lower) & (x_raw <= upper)).to(torch.uint8)
                                expected_h_mask = ((h_raw >= lower) & (h_raw <= upper)).to(torch.uint8)
                                torch.testing.assert_close(result[7], expected_x_mask, rtol=0, atol=0)
                                torch.testing.assert_close(result[8], expected_h_mask, rtol=0, atol=0)
                                arguments['is_training'] = False
                                evaluated = self.forward(carrier, arguments)
                                torch.testing.assert_close(evaluated[6], result[6], rtol=0, atol=0)
                                torch.testing.assert_close(evaluated[0], result[0], rtol=0, atol=0)

    def test_real_gate_half_steps_round_before_zero_point(self):
        if not quant_gru.gru_ops.use_real_activation:
            self.skipTest('Exact standard-activation half steps require a real-activation build')
        for carrier in ('float32', 'int32'):
            for bits in (8, 16):
                for pot2 in (False, True):
                    for zero_point in (-3, 3):
                        with self.subTest(carrier=carrier, bits=bits,
                                          pot2=pot2, zero_point=zero_point):
                            arguments, parameters = self.fixture(bits, pot2)
                            device = arguments['x'].device
                            arguments['batch_size'] = 3
                            arguments['W'].fill_(1)
                            arguments['x'] = torch.tensor([-16., 0., 16.], device=device).reshape(1, 3, 1)
                            arguments['h0'] = torch.zeros(3, 1, device=device)
                            for gate in ('update', 'reset', 'new'):
                                setattr(parameters, 'zp_' + gate + '_gate_output_', zero_point)
                            parameters.scale_new_gate_output_ = 2.0
                            result = self.forward(carrier, arguments)
                            cache = result[1][0]
                            expected_sigmoid = torch.tensor([0., 0., 1.], device=device) + zero_point
                            expected_sigmoid_mask = ((expected_sigmoid >= 0) &
                                                     (expected_sigmoid <= (1 << bits) - 1)).to(torch.uint8)
                            expected_sigmoid = expected_sigmoid.clamp(0, (1 << bits) - 1)
                            for index in (0, 1):
                                actual_codes = cache[:, index] + zero_point
                                torch.testing.assert_close(actual_codes, expected_sigmoid, rtol=0, atol=0)
                                if not quant_gru._CPU_ONLY:
                                    torch.testing.assert_close(result[16][0, :, index],
                                                               expected_sigmoid_mask, rtol=0, atol=0)
                            expected_tanh = torch.full((3,), float(zero_point), device=device)
                            actual_tanh = cache[:, 2] / 2.0 + zero_point
                            torch.testing.assert_close(actual_tanh, expected_tanh, rtol=0, atol=0)
                            if not quant_gru._CPU_ONLY:
                                torch.testing.assert_close(result[16][0, :, 2],
                                                           torch.ones(3, device=device, dtype=torch.uint8),
                                                           rtol=0, atol=0)
                                arguments['is_training'] = False
                                evaluated = self.forward(carrier, arguments)
                                torch.testing.assert_close(evaluated[0], result[0], rtol=0, atol=0)

    def test_input_saturates_before_integer_conversion_and_records_mask(self):
        widths = ((8, False, 127), (16, False, 32767),
                  (26, False, 33554430), (32, False, 2147483520),
                  (31, True, 2147483520))
        for carrier in ('float32', 'int32'):
            for bits, unsigned, float_upper in widths:
                for pot2 in (False, True):
                    with self.subTest(carrier=carrier, bits=bits,
                                      unsigned=unsigned, pot2=pot2):
                        arguments, parameters = self.fixture(8, pot2)
                        config = parameters.bitwidth_config_
                        config.x_ = bits
                        config.x_unsigned_ = unsigned
                        parameters.bitwidth_config_ = config
                        parameters.scale_x_ = .5 if pot2 else 1.0
                        device = arguments['x'].device
                        lower = 0 if unsigned else -(1 << (bits - 1))
                        upper = (1 << bits) - 1 if unsigned else (1 << (bits - 1)) - 1
                        lo = torch.tensor(float(lower), device=device)
                        hi = torch.tensor(float(upper), device=device)
                        negative_inf = torch.tensor(float('-inf'), device=device)
                        positive_inf = torch.tensor(float('inf'), device=device)
                        codes = torch.stack([
                            torch.tensor(float(-(1 << 32)), device=device),
                            torch.nextafter(lo, negative_inf), lo,
                            torch.nextafter(lo, positive_inf), lo - 1,
                            torch.tensor(-1., device=device),
                            torch.tensor(0., device=device),
                            torch.tensor(1., device=device),
                            torch.nextafter(hi, negative_inf), hi,
                            torch.nextafter(hi, positive_inf), hi + 1,
                            torch.tensor(float(1 << 32), device=device),
                        ])
                        arguments['batch_size'] = codes.numel()
                        arguments['x'] = (codes * parameters.scale_x_).reshape(1, -1, 1)
                        arguments['h0'] = torch.zeros(codes.numel(), 1, device=device)
                        result = self.forward(carrier, arguments)

                        rounded = torch.round(codes).double().reshape(1, -1, 1)
                        expected_int = rounded.clamp(lower, upper).to(torch.int32)
                        if carrier == 'float32' and not quant_gru._CPU_ONLY:
                            # FP32 stores the greatest representable code inside the integer domain.
                            expected = rounded.clamp(lower, float_upper).float()
                            self.assertTrue((result[6].double() <= upper).all())
                            self.assertTrue((result[6].double() >= lower).all())
                        else:
                            # CPU float storage is a cast of its integer core's cached codes.
                            expected = expected_int.float() if carrier == 'float32' else expected_int
                        torch.testing.assert_close(result[6], expected, rtol=0, atol=0)
                        self.assertTrue(torch.isfinite(result[0]).all())
                        if not quant_gru._CPU_ONLY:
                            expected_mask = ((rounded >= lower) & (rounded <= upper)).to(torch.uint8)
                            torch.testing.assert_close(result[7], expected_mask, rtol=0, atol=0)
                            arguments['is_training'] = False
                            evaluated = self.forward(carrier, arguments)
                            torch.testing.assert_close(evaluated[6], result[6], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
