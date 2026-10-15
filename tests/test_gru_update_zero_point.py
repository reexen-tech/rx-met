"""Changing an unclipped gate's affine origin must not change GRU outputs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from tests.test_gru_clamp_chain_rule import fixed_model


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class GruUpdateZeroPointTest(unittest.TestCase):
    def test_equivalent_update_gate_grids(self):
        for carrier in ('float32', 'int32'):
            for pot2 in (False, True):
                for steps in (1, 7):
                    with self.subTest(carrier=carrier, pot2=pot2, steps=steps), tempfile.TemporaryDirectory() as directory:
                        model, original = fixed_model(directory)
                        model.quant_storage_dtype = carrier
                        x = torch.full((steps, 2, 1), .5, device='cuda')
                        reference = None
                        for zero_point in (0, 1, 8, 16, 64):
                            data = copy.deepcopy(original)
                            data['model_info']['use_pot2_scale'] = pot2
                            gate = data['operators']['update_gate_output']
                            gate.update(symmetric=False, zero_point=zero_point,
                                real_min=-zero_point*gate['scale'], real_max=(255-zero_point)*gate['scale'])
                            path = Path(directory) / 'shifted.json'
                            path.write_text(json.dumps(data))
                            model.load_quant_params(str(path))
                            y, hidden = model(x)
                            saved = y.grad_fn.saved_tensors
                            self.assertTrue(torch.all(saved[16][..., 0] == 1))
                            if reference is None:
                                reference = (y.detach(), hidden.detach())
                            else:
                                torch.testing.assert_close(y, reference[0], atol=0, rtol=0)
                                torch.testing.assert_close(hidden, reference[1], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
