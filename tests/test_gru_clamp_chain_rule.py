"""Native forward masks + independent autograd surrogate for GRU QAT."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from quant_gru import QuantGRU


def swap_gates(t):
    r,z,n=t.chunk(3,dim=0)
    return torch.cat((z,r,n),dim=0)


def fixed_model(directory, clipped=None):
    m=QuantGRU(1,1).cuda();m.set_all_bitwidth(8)
    with torch.no_grad():
        m.weight_ih_l0.copy_(torch.tensor([[.2],[.1],[.3]],device='cuda'))
        m.weight_hh_l0.copy_(torch.tensor([[.1],[.1],[.2]],device='cuda'))
        m.bias_ih_l0.copy_(torch.tensor([.1,.1,.2],device='cuda'))
        m.bias_hh_l0.copy_(torch.tensor([.1,.1,.4],device='cuda'))
    m.calibrating=True
    with torch.no_grad():m(torch.tensor([[[.5]]],device='cuda'))
    m.calibrating=False;m.finalize_calibration()
    path=Path(directory)/'encodings.json';m.export_quant_params(str(path));data=json.loads(path.read_text())
    data['model_info']['use_pot2_scale']=True
    for name,e in data['operators'].items():
        scale=1/32 if name.endswith('_input') or name.endswith('_linear') else 1/128
        if name in ('update_gate_output','reset_gate_output'):scale=1/256
        if name==clipped:scale=1/4096
        channels=len(e['scale']) if isinstance(e['scale'],list) else None
        lo,hi=(0,255) if e['dtype'].startswith('U') else (-128,127)
        e.update(scale=[scale]*channels if channels else scale,zero_point=[0]*channels if channels else 0,
                 symmetric=True,real_min=[scale*lo]*channels if channels else scale*lo,
                 real_max=[scale*hi]*channels if channels else scale*hi)
    path.write_text(json.dumps(data));m.load_quant_params(str(path));m.use_quantization=True;m.train()
    return m,data


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

if __name__=='__main__':unittest.main()
