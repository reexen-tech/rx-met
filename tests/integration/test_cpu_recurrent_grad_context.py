"""CPU QuantSim inference accepts the default PyTorch gradient context."""
import contextlib
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn
import gru_interface_binding
from quant_gru import QuantGRU
from quant_lstm import QuantLSTM

from aimet_common import run_context
from aimet_torch.model_preparer import prepare_model
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU, QuantizedQuantLSTM
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel
from tests.support.paths import REPO_ROOT


class ForwardingGRU(QuantGRU):
    def forward(self, input, hx=None):
        return super().forward(input, hx)


class ForwardingLSTM(QuantLSTM):
    def forward(self, input, hx=None):
        return super().forward(input, hx)


@QuantizationMixin.implements(ForwardingGRU)
class QuantizedForwardingGRU(QuantizedQuantGRU, ForwardingGRU):
    pass


@QuantizationMixin.implements(ForwardingLSTM)
class QuantizedForwardingLSTM(QuantizedQuantLSTM, ForwardingLSTM):
    pass


class RecurrentModel(nn.Module):
    def __init__(self, kind, *, bidirectional, mixed):
        super().__init__()
        self.mixed = mixed
        if mixed:
            self.proj = nn.Linear(3, 3)
            self.head = nn.Linear(8 if bidirectional else 4, 2)
        operator = ForwardingGRU if kind == 'gru' else ForwardingLSTM
        self.recurrent = operator(3, 4, batch_first=True, bidirectional=bidirectional)

    def forward(self, value):
        if self.mixed:
            value = self.proj(value)
        sequence, _ = self.recurrent(value)
        return self.head(sequence) if self.mixed else sequence


@unittest.skipUnless(getattr(gru_interface_binding, 'cpu_only', False), 'CPU wheels required')
class CpuRecurrentGradContextTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cpu-recurrent-grad-')
        self.addCleanup(temporary.cleanup)
        run_context.configure_output(timestamp_output=False)
        self.addCleanup(run_context.configure_output)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(run_context.execution(Path(temporary.name)))
        stack.enter_context(torch.enable_grad())
        torch.manual_seed(53)
        self.value = torch.linspace(-0.7, 0.9, 30).reshape(2, 5, 3)

    def simulation(self, kind, bidirectional, *, mixed=False):
        model = RecurrentModel(kind, bidirectional=bidirectional, mixed=mixed).eval()
        prepared = prepare_model(model)
        simulation = QuantizationSimModel(
            prepared, self.value, quant_scheme='tf',
            config_file=str(REPO_ROOT / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'))
        apply_mixed_precision_bitwidth(
            simulation.model, str(REPO_ROOT / 'examples/config/quick_start_full_quant.json'), verbose=False)
        simulation.model.eval()
        return simulation

    def assert_default_context_forward(self, simulation):
        self.assertTrue(torch.is_grad_enabled())
        actual = simulation.model(self.value)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(actual.requires_grad)
        with torch.no_grad():
            reference = simulation.model(self.value)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, r'CPU Quant(?:GRU|LSTM).*backward'):
            actual.sum().backward()

    def test_context_calibration_and_direct_inference(self):
        for kind in ('gru', 'lstm'):
            for bidirectional in (False, True):
                with self.subTest(kind=kind, bidirectional=bidirectional):
                    simulation = self.simulation(kind, bidirectional)
                    with compute_encodings(simulation.model):
                        simulation.model(self.value)
                        simulation.model(self.value * 0.8)
                    self.assertTrue(simulation.model.recurrent.is_calibrated())
                    self.assertTrue(simulation.model.recurrent.use_quantization)
                    self.assert_default_context_forward(simulation)

    def test_callback_calibration_restores_default_grad_context(self):
        for kind in ('gru', 'lstm'):
            with self.subTest(kind=kind):
                simulation = self.simulation(kind, True)
                simulation.compute_encodings(lambda model, value: model(value), self.value)
                self.assert_default_context_forward(simulation)

    def test_mixed_quantized_network_cannot_silently_drop_gradients(self):
        for kind in ('gru', 'lstm'):
            with self.subTest(kind=kind):
                simulation = self.simulation(kind, True, mixed=True)
                simulation.compute_encodings(lambda model, value: model(value), self.value)
                self.assert_default_context_forward(simulation)


if __name__ == '__main__':
    unittest.main()
