"""Fixed bias grids must permit small QAT updates, including zero channels."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from quant_gru import QuantGRU
from quant_lstm import QuantLSTM


def export(model,path):
    if isinstance(model,QuantLSTM):return model.export_quant_params()
    model.export_quant_params(str(path));return json.loads(path.read_text())


@unittest.skipUnless(torch.cuda.is_available(),'CUDA is required')
class RecurrentBiasHeadroomTest(unittest.TestCase):
    def test_updates_preserve_fixed_encoding_and_nonzero_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'encodings.json'
            for cls in (QuantGRU,QuantLSTM):
                for bits in (8,16):
                    for pot2 in (False,True):
                        for initial,step in ((0.,1e-4),(.2,.05),(-.2,-.05)):
                            with self.subTest(operator=cls.__name__,bits=bits,pot2=pot2,initial=initial):
                                torch.manual_seed(42)
                                model=cls(4,4).cuda();model.set_all_bitwidth(bits)
                                if pot2:model.enable_pot2()
                                with torch.no_grad():model.bias_ih_l0.fill_(initial);model.bias_hh_l0.fill_(initial)
                                x=torch.randn(5,2,4,device='cuda')*.5
                                model.calibrating=True
                                with torch.no_grad():model(x)
                                model.calibrating=False;model.finalize_calibration();model.use_quantization=True
                                before=export(model,path);model.set_quant_params_locked(True)
                                enc=before['operators']['bias_ih']
                                scales=torch.as_tensor(enc['scale'],device='cuda')
                                new_value=initial+step
                                self.assertTrue(torch.all(torch.round(torch.tensor(new_value,device='cuda')/scales)<=2**(bits-1)-1))
                                self.assertTrue(torch.all(torch.round(torch.tensor(new_value,device='cuda')/scales)>=-2**(bits-1)))
                                if initial==0:self.assertTrue(torch.all(torch.round(torch.tensor(step,device='cuda')/scales)!=0))
                                with torch.no_grad():model.bias_ih_l0.add_(step);model.bias_hh_l0.add_(step)
                                model.train();y,_=model(x);y.square().mean().backward()
                                self.assertGreater(model.bias_ih_l0.grad.abs().max().item(),0.)
                                self.assertEqual(before,export(model,path))
                                clone=cls(4,4).cuda();clone.load_state_dict(model.state_dict())
                                if isinstance(clone,QuantLSTM):clone.load_quant_params(before)
                                else:path.write_text(json.dumps(before));clone.load_quant_params(str(path))
                                clone.use_quantization=True
                                model.eval();clone.eval()
                                with torch.no_grad():torch.testing.assert_close(clone(x)[0],model(x)[0],rtol=0,atol=0)

    def test_zero_bias_all_calibration_methods_and_granularities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encodings.json"
            for cls in (QuantGRU, QuantLSTM):
                for method in ("minmax", "percentile", "sqnr"):
                    for granularity in ("per_tensor", "per_gate", "per_channel"):
                        with self.subTest(operator=cls.__name__, method=method, granularity=granularity):
                            torch.manual_seed(42)
                            model = cls(4, 4).cuda()
                            model.calibration_method = method
                            if cls is QuantGRU:
                                index = ("per_tensor", "per_gate", "per_channel").index(granularity)
                                model._bitwidth_config.bw_granularity_ = index
                                model._bitwidth_config.br_granularity_ = index
                            else:
                                for name in ("bias_ih", "bias_hh"):
                                    model.adjust_quant_config(name, granularity=granularity)
                            with torch.no_grad():
                                model.bias_ih_l0.zero_()
                                model.bias_hh_l0.zero_()
                                model.calibrating = True
                                model(torch.randn(5, 2, 4, device="cuda") * .5)
                            model.calibrating = False
                            model.finalize_calibration()
                            encodings = export(model, path)
                            for name in ("bias_ih", "bias_hh"):
                                scales = torch.as_tensor(encodings["operators"][name]["scale"])
                                codes = torch.round(1e-4 / scales)
                                self.assertTrue(torch.all(codes > 0))
                                self.assertTrue(torch.all(codes <= 127))

if __name__=='__main__':unittest.main()
