"""Export biased Linear as one fused compute op at every input rank."""
from types import MethodType

import onnx
import torch.nn as nn
import torch.nn.functional as F


def _forward(self, input):
    if input.dim() == 2:
        return F.linear(input, self.weight, self.bias)
    # Reshapes preserve the input/output quantization grids. Keeping the bias
    # inside Gemm avoids inventing a grid for a MatMul -> Add intermediate.
    output = F.linear(input.reshape(-1, self.in_features), self.weight, self.bias)
    return output.reshape(*input.shape[:-1], self.out_features)


def prepare_linear_export(model):
    """Adapt only the disposable unquantized export copy, retaining parameters.

    Bias-free Linear already exports to one MatMul for arbitrary input ranks.
    Do not override subclasses that may implement different forward semantics.
    """
    names = []
    for name, module in model.named_modules():
        if type(module) is nn.Linear and module.bias is not None:
            module.forward = MethodType(_forward, module)
            names.append(name)
    return names


def restore_linear_names(onnx_path, names):
    """Assign each adapted module's encoding key to its sole Gemm node.

    AIMET markers associate all surrounding shape operations with the same
    module. Swap names if the input Reshape currently owns the unsuffixed name.
    No encoding values or tensor/parameter names are changed here.
    """
    graph = onnx.load(onnx_path)
    changed = False
    for name in names:
        nodes = [node for node in graph.graph.node
                 if node.name == name or node.name.startswith(name + '#')]
        compute = [node for node in nodes if node.op_type == 'Gemm']
        if not compute:
            continue  # An unused module may be absent from the traced graph.
        if len(compute) != 1:
            raise RuntimeError(f'Linear {name}: expected one fused Gemm, found {len(compute)}')
        target = compute[0]
        if target.name != name:
            previous = target.name
            for node in nodes:
                if node.name == name:
                    node.name = previous
            target.name = name
            changed = True
    if changed:
        onnx.save(graph, onnx_path)
