"""Native GRU clamp derivatives, train/eval autograd and affine gate origins."""

import json
from pathlib import Path
import tempfile
import unittest
import torch
from quant_gru import QuantGRU
from unittest import mock
import quant_gru
import copy

from tests.support.recurrent import fixed_gru_model as fixed_model


def swap_gates(t):
    r,z,n=t.chunk(3,dim=0)
    return torch.cat((z,r,n),dim=0)


def surrogate(saved,enc):
    """Reconstruct derivatives in real units; values are anchored to native QDQ.

    Every clamp masks its immediate input before autograd propagates to parents.
    Gate derivatives use dequantized outputs, the public STE convention.
    """
    wq,rq,bwq,brq,xq,h,v,xmask,h0mask,wmask,rmask,bwmask,brmask,pmask,qmask,imask,omask,hmask=saved
    def leaf(q,key):
        s=torch.tensor(enc['operators'][key]['scale'],device=q.device)
        return (q*s).detach().requires_grad_()
    w=leaf(wq,'weight_ih');r=leaf(rq,'weight_hh');bw=leaf(bwq,'bias_ih');br=leaf(brq,'bias_hh');x=leaf(xq,'input')
    h0=h[0].detach().clone().requires_grad_()
    def masked(value,mask):return value.detach()+(value-value.detach())*mask
    def anchor(value,actual):return actual.detach()+(value-value.detach())
    mw=masked(w,wmask);mr=masked(r,rmask);mbw=masked(bw,bwmask);mbr=masked(br,brmask);mx=masked(x,xmask)
    prev=masked(h0,h0mask);ys=[]
    for t in range(x.shape[0]):
        p=masked(mx[t]@mw+mbw,pmask[t]);q=masked(prev@mr+mbr,qmask[t])
        z0,r0,n0,qg=v[t].chunk(4,dim=-1)
        iz,ir,ig=imask[t].chunk(3,dim=-1);oz,orr,og=omask[t].chunk(3,dim=-1)
        pz,pr,pg=p.chunk(3,dim=-1);qz,qr,qn=q.chunk(3,dim=-1)
        def activation(pre,out,mask,derivative):
            return out.detach()+(pre-pre.detach())*mask*derivative.detach()
        z=activation(pz+qz,z0,iz*oz,z0*(1-z0))
        reset=activation(pr+qr,r0,ir*orr,r0*(1-r0))
        candidate=activation(pg+reset*anchor(qn,qg),n0,ig*og,1-n0*n0)
        prev=anchor(masked(z*prev+(1-z)*candidate,hmask[t]),h[t+1]);ys.append(prev)
    return torch.stack(ys),prev.unsqueeze(0),(w,r,bw,br,x,h0)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA is required')
class GruClampChainRuleTest(unittest.TestCase):
    def check_case(self,clipped,steps,batch,loss_source):
        with tempfile.TemporaryDirectory() as directory:
            m,enc=fixed_model(directory,clipped)
            x=torch.linspace(.35,.65,steps*batch,device='cuda').reshape(steps,batch,1).requires_grad_()
            h0=torch.full((1,batch,1),.2,device='cuda',requires_grad=True)
            y,hn=m(x,h0);saved=tuple(t.detach().clone() for t in y.grad_fn.saved_tensors)
            reference,reference_hn,leaves=surrogate(saved,enc)
            if clipped in ('new_gate_input','new_gate_output'):
                mask=saved[15 if clipped=='new_gate_input' else 16][...,2]
                self.assertTrue(torch.all(mask==0))
            if clipped=='weight_ih_linear':self.assertTrue(torch.all(saved[13]==0))
            def loss(output,last):
                weights=torch.linspace(.3,1.1,output.numel(),device='cuda').reshape_as(output)
                if loss_source=='output':return (output*weights).sum()
                if loss_source=='hidden':return last.square().sum()
                return (output*weights).sum()+last.square().sum()
            loss(y,hn).backward();loss(reference,reference_hn).backward()
            actual=[swap_gates(m.weight_ih_l0.grad).t(),swap_gates(m.weight_hh_l0.grad).t(),
                    swap_gates(m.bias_ih_l0.grad),swap_gates(m.bias_hh_l0.grad),x.grad,h0.grad.squeeze(0)]
            for name,a,b in zip(['W','R','bias_ih','bias_hh','input','h0'],actual,leaves):
                torch.testing.assert_close(a,b.grad,atol=1e-6,rtol=1e-5,msg=lambda s:f'{name}: {s}')

    def test_unclipped_control(self):
        for steps,batch in [(1,1),(4,2)]:
            for loss in ['output','hidden','both']:
                with self.subTest(steps=steps,batch=batch,loss=loss):self.check_case(None,steps,batch,loss)

    def test_candidate_clamp_blocks_reset_path(self):
        for clipped in ['new_gate_input','new_gate_output']:
            for steps,batch in [(1,1),(4,2)]:
                for loss in ['output','hidden','both']:
                    with self.subTest(clipped=clipped,steps=steps,batch=batch,loss=loss):self.check_case(clipped,steps,batch,loss)

    def test_input_linear_clamp_blocks_bias_path(self):
        for steps,batch in [(1,1),(4,2)]:
            for loss in ['output','hidden','both']:
                with self.subTest(steps=steps,batch=batch,loss=loss):self.check_case('weight_ih_linear',steps,batch,loss)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class GruEvalBackwardTest(unittest.TestCase):
    def backward(self, model, value, h0):
        model.zero_grad(set_to_none=True)
        value = value.detach().clone().requires_grad_()
        h0 = h0.detach().clone().requires_grad_()
        y, last = model(value, h0)
        pending, visited, saved = [y.grad_fn], set(), []
        while pending:
            node = pending.pop()
            if node is None or node in visited:
                continue
            visited.add(node)
            if type(node).__name__ == 'GRUFunctionBackward':
                saved.extend(t.detach().clone() for t in node.saved_tensors)
            pending.extend(parent for parent, _ in node.next_functions)
        self.assertTrue(saved)
        (y.square().sum() + last.sum()).backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                     if p.requires_grad}
        gradients.update(input=value.grad, hidden=h0.grad)
        return (y.detach(), last.detach()), saved, gradients

    def check_modes(self, model, x, h):
        model.train()
        expected, cache, gradients = self.backward(model, x, h)
        model.eval()
        actual, actual_cache, actual_gradients = self.backward(model, x, h)
        self.assertFalse(model.training)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for index, (left, right) in enumerate(zip(cache, actual_cache)):
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=f'缓存 {index}')
        for name in gradients:
            torch.testing.assert_close(gradients[name], actual_gradients[name], rtol=0, atol=0,
                                       msg=f'梯度 {name}')

    def test_eval_saturation_uses_the_same_masks(self):
        for carrier in ['float32', 'int32']:
            for clipped in [None, 'weight_ih_linear', 'new_gate_input', 'new_gate_output']:
                with self.subTest(carrier=carrier, clipped=clipped), tempfile.TemporaryDirectory() as directory:
                    model, _ = fixed_model(directory, clipped)
                    model.quant_storage_dtype = carrier
                    x = torch.tensor([[[.5]], [[.4]]], device='cuda')
                    h = torch.tensor([[[.2]]], device='cuda')
                    self.check_modes(model, x, h)

    def test_eval_floating_and_8_16_bit_bidirectional_backward(self):
        for bits in [None, 8, 16]:
            with self.subTest(bits=bits):
                torch.manual_seed(42)
                model = QuantGRU(3, 5, bidirectional=True).cuda()
                # 单样本避免 bias 的 atomicAdd 并发归约顺序差异，
                # 使零容差断言只检查 train/eval 的缓存和反向语义。
                x = torch.randn(4, 1, 3, device='cuda') * .2
                h = torch.randn(2, 1, 5, device='cuda') * .1
                if bits is not None:
                    model.set_all_bitwidth(bits)
                    model.calibrating = True
                    with torch.no_grad():
                        model(x, h)
                    model.calibrating = False
                    model.finalize_calibration()
                    model.use_quantization = True
                    model.set_quant_params_locked(True)
                self.check_modes(model, x, h)

    def test_eval_with_frozen_parameters_still_supports_input_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _ = fixed_model(directory, 'weight_ih_linear')
            model.requires_grad_(False)
            self.check_modes(model, torch.tensor([[[.5]]], device='cuda'),
                             torch.tensor([[[.2]]], device='cuda'))

    def test_eval_parameter_gradients_without_input_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _ = fixed_model(directory, 'weight_ih_linear')
            model.eval()
            y, _ = model(torch.tensor([[[.5]]], device='cuda'))
            self.assertGreater(y.grad_fn.saved_tensors[13].numel(), 0)
            y.sum().backward()
            torch.testing.assert_close(model.bias_ih_l0.grad,
                                       torch.zeros_like(model.bias_ih_l0), rtol=0, atol=0)

    def test_eval_no_grad_keeps_inference_cache_path(self):
        for carrier, function in [('float32', 'forward_quant_float_storage'),
                                  ('int32', 'forward_quant_int_storage')]:
            with self.subTest(carrier=carrier), tempfile.TemporaryDirectory() as directory:
                model, _ = fixed_model(directory)
                model.eval()
                model.quant_storage_dtype = carrier
                original = getattr(quant_gru.gru_ops, function)
                with mock.patch.object(quant_gru.gru_ops, function, wraps=original) as call:
                    with torch.no_grad():
                        y, _ = model(torch.tensor([[[.5]]], device='cuda'))
                self.assertFalse(call.call_args.kwargs['is_training'])
                self.assertFalse(y.requires_grad)


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


if __name__ == "__main__":
    unittest.main()
