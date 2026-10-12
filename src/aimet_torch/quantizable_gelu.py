"""GELU expressed as elementary modules before calibration and QAT."""
import math

from torch import nn

from aimet_torch._base.nn.modules.custom import Add, Divide, Erf, Multiply, Pow


class QuantizableGELU(nn.Module):
    """Preserve PyTorch's approximation while exposing every compute operand.

    Only the selected approximation's modules are created: unused quantizers
    must not appear as uncalibrated entries during strict export validation.
    """

    def __init__(self, approximate='none'):
        super().__init__()
        if approximate not in ('none', 'tanh'):
            raise ValueError(f'Unsupported GELU approximation: {approximate!r}')
        self.approximate = approximate
        if approximate == 'none':
            self.divide = Divide()
            self.erf = Erf()
        else:
            self.cube = Pow()
            self.cubic_scale = Multiply()
            self.inner_add = Add()
            self.normal_scale = Multiply()
            self.tanh = nn.Tanh()
        self.add = Add()
        self.half_scale = Multiply()
        self.multiply = Multiply()

    def forward(self, input):
        if self.approximate == 'none':
            normalized = self.divide(input, math.sqrt(2.0))
            activation = self.erf(normalized)
        else:
            cubic = self.cube(input, 3.0)
            scaled_cubic = self.cubic_scale(cubic, 0.044715)
            inner = self.inner_add(input, scaled_cubic)
            normalized = self.normal_scale(inner, math.sqrt(2.0 / math.pi))
            activation = self.tanh(normalized)
        shifted = self.add(activation, 1.0)
        half = self.half_scale(input, 0.5)
        return self.multiply(half, shifted)
