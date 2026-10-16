"""Parameter quantization for modules expanded into FX get_attr operations."""
import contextlib

import torch
from torch import nn

from aimet_torch.v2.quantization.base import QuantizerBase


class ParameterQuantizers(nn.ModuleDict):
    """Keep the public parameter-quantizer mapping callable from an FX graph."""

    def forward(self, name, value):
        quantizer = self[name]
        return quantizer(value) if quantizer is not None else value


_VIEW_METHODS = {'view', 'reshape', 'flatten', 'transpose', 'permute',
                 'contiguous', 'unsqueeze', 'squeeze', 'expand', 'expand_as'}
_VIEW_FUNCTIONS = {torch.reshape, torch.flatten, torch.transpose, torch.permute,
                   torch.unsqueeze, torch.squeeze}
_NODE_TAG = 'rx_parameter_quantizer'


def bind_parameter_quantizers(model):
    """Insert one QDQ at parameter reads and remove duplicate consumer QDQ."""
    for graph_module in model.modules():
        if not isinstance(graph_module, torch.fx.GraphModule):
            continue
        parameter_values = set()
        for node in list(graph_module.graph.nodes):
            if node.op == 'get_attr' and '.' in node.target:
                owner_name, param_name = node.target.rsplit('.', 1)
                owner = graph_module.get_submodule(owner_name)
                if not getattr(owner, '_param_quantization_enabled', False):
                    continue
                if param_name not in owner.param_quantizers:
                    continue
                with graph_module.graph.inserting_after(node):
                    quantized = graph_module.graph.call_module(
                        owner_name + '.param_quantizers', (param_name, node))
                quantized.meta[_NODE_TAG] = True
                node.replace_all_uses_with(quantized)
                quantized.args = (param_name, node)
                parameter_values.add(quantized)
            elif ((node.op == 'call_method' and node.target in _VIEW_METHODS)
                  or (node.op == 'call_function' and node.target in _VIEW_FUNCTIONS)):
                if node.args and node.args[0] in parameter_values:
                    parameter_values.add(node)
            elif node.op == 'call_module':
                module = graph_module.get_submodule(node.target)
                quantizers = getattr(module, 'input_quantizers', [])
                for index, arg in enumerate(node.args):
                    if (isinstance(arg, torch.fx.Node) and arg in parameter_values
                            and index < len(quantizers)):
                        quantizers[index] = None
        graph_module.graph.lint()
        graph_module.recompile()


def remove_parameter_quantizer_nodes(model, excluded_modules):
    """Return the original FX reads when constructing an unquantized model."""
    if not isinstance(model, torch.fx.GraphModule):
        return
    changed = False
    for node in list(model.graph.nodes):
        if (node.meta.get(_NODE_TAG)
                and model.get_submodule(node.target.rsplit('.', 1)[0]) in excluded_modules):
            node.replace_all_uses_with(node.args[1])
            model.graph.erase_node(node)
            changed = True
    if changed:
        model.graph.lint()
        model.recompile()


@contextlib.contextmanager
def compute_parameter_encodings(module):
    """Observe the FP parameters during calibration, with QDQ in passthrough."""
    from aimet_torch.v2.utils import patch_attr

    with contextlib.ExitStack() as stack:
        for quantizer in module.param_quantizers.values():
            if isinstance(quantizer, QuantizerBase) and quantizer._allow_overwrite:
                # Match BaseQuantizationMixin.compute_encodings: observe the
                # master parameter, but don't quantize the calibration graph.
                stack.enter_context(patch_attr(quantizer, 'forward', lambda x: x))
                stack.enter_context(quantizer.compute_encodings())
        yield


@torch.no_grad()
def update_parameter_encodings(module):
    """Calibrate only the expanded owner's parameters, without model inputs."""
    with compute_parameter_encodings(module):
        for name, quantizer in module.param_quantizers.items():
            if isinstance(quantizer, QuantizerBase) and quantizer._allow_overwrite:
                quantizer(getattr(module, name))
