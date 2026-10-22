"""Exercise the public PyTorch Po2 workflow with real calibrated quantizers."""
from pathlib import Path
import unittest
import torch
from torch import nn
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth, apply_power_of_2_workflow


class BiasedLayer(nn.Module):
    def __init__(self, convolution=False):
        super().__init__()
        self.layer = nn.Conv2d(4,2,1) if convolution else nn.Linear(4,2)
        with torch.no_grad():
            self.layer.weight.fill_(.01)
            self.layer.bias.copy_(torch.tensor([.5,-.25]))

    def forward(self,x):
        return self.layer(x)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA is required')
class BiasAlignmentRangeTest(unittest.TestCase):
    def calibrated(self,convolution=False):
        root=Path('/repo') if Path('/repo').exists() else Path(__file__).resolve().parents[1]
        x=torch.tensor([[.1,-.1,.1,-.1]],device='cuda')
        if convolution:x=x.reshape(1,4,1,1)
        model=BiasedLayer(convolution).cuda().eval()
        expected=model(x).detach()
        sim=QuantizationSimModel(prepare_model(model),x,quant_scheme='tf',
            config_file=str(root/'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'))
        apply_mixed_precision_bitwidth(sim.model,str(root/'examples/config/quick_start_full_quant.json'),verbose=False)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(x)
            # Keep the output grid away from its own clipping boundary, so this
            # regression isolates bias alignment rather than Po2 output clipping.
            sim.model(x + 1.0)
        return sim,x,expected

    def test_requested_bias32_preserves_representable_bias(self):
        for convolution in [False,True]:
            with self.subTest(convolution=convolution):
                sim,x,expected=self.calibrated(convolution)
                apply_power_of_2_workflow(sim.model,align_bias_scale=True,bias_bitwidth=32,verbose=False)
                layer=sim.model.layer;q=layer.param_quantizers['bias']
                self.assertEqual(q.bitwidth,32)
                product=layer.input_quantizers[0].get_scale()*layer.param_quantizers['weight'].get_scale().reshape(-1)
                torch.testing.assert_close(q.get_scale().reshape(-1),product,rtol=0,atol=0)
                torch.testing.assert_close(sim.model(x),expected,rtol=0,atol=1e-7)

    def test_explicit_bias16_rejects_unrepresentable_values(self):
        sim,_,_=self.calibrated()
        with self.assertRaisesRegex(ValueError,'bias.*represent|bias.*范围'):
            apply_power_of_2_workflow(sim.model,align_bias_scale=True,bias_bitwidth=16,verbose=False)

    def test_bias32_still_rejects_overflow(self):
        sim,_,_=self.calibrated()
        with torch.no_grad():sim.model.layer.bias.fill_(1e12)
        with self.assertRaisesRegex(ValueError,'bias.*represent|bias.*范围'):
            apply_power_of_2_workflow(sim.model,align_bias_scale=True,bias_bitwidth=32,verbose=False)

if __name__=='__main__':unittest.main()
