"""Short-lived hierarchical traces used only to inspect a ConnectedGraph."""

import torch


class _ForwardOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.training = model.training

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def trace_for_connected_graph(model, model_input, **trace_args):
    """Trace without registering disposable module types in Torch's global CU.

    Ordinary ``jit.trace`` creates fresh module types in a process-wide
    CompilationUnit and stores their concrete types in a strong global set.
    ConnectedGraph needs the hierarchy only while parsing it, so give that
    hierarchy its own CompilationUnit. Parameters remain tensor attributes in
    this analysis-only trace; ConnectedGraph reads parameters from ``model``.
    The result is not a replacement for an executable/exportable ScriptModule.
    """
    if trace_args.pop("_module_class", None) is not None:
        raise ValueError(
            "ConnectedGraph tracing does not support a custom _module_class; "
            "its analysis modules must use a local CompilationUnit."
        )
    if not isinstance(model, torch.jit.ScriptModule) and (
        torch._jit_internal.module_has_exports(model)
    ):
        # Torch scripts exported root methods before consulting _module_class.
        # Only forward is relevant to CG; keep it below an unexported root so
        # the disposable hierarchy always uses the local CompilationUnit.
        return torch.jit.trace(
            _ForwardOnly(model),
            model_input,
            _module_class=_make_transient_module,
            **trace_args,
        ).model
    return torch.jit.trace(
        model, model_input, _module_class=_make_transient_module, **trace_args
    )


def _make_transient_module(model, _compilation_unit=None):
    # Each native module type and its traced methods belong to this CU. No
    # process-global Torch caches are replaced or cleared.
    compilation_unit = torch.jit.CompilationUnit()
    converted = {}

    def convert(module):
        if isinstance(module, torch.jit.ScriptModule):
            return module._c
        if module in converted:
            return converted[module]
        if module._backward_hooks:
            raise ValueError(
                "Modules that have backward hooks assigned can't be compiled: "
                + str(module)
            )

        native = torch._C.ScriptModule(
            torch._jit_internal._qualified_name(type(module)),
            compilation_unit,
            True,
        )
        converted[module] = native
        native._register_attribute(
            "training", torch._C.BoolType.get(), module.training
        )
        for name, tensor in {**module._parameters, **module._buffers}.items():
            if tensor is not None:
                native._register_attribute(name, torch._C.TensorType.get(), tensor)
        for name, value in module.__dict__.items():
            if (
                name not in module._parameters
                and name not in module._buffers
                and torch._C._jit_is_script_object(value)
            ):
                value_type = torch._C._jit_try_infer_type(value).type()
                native._register_attribute(name, value_type, value)
        for name, child in module._modules.items():
            if child is not None:
                native_child = convert(child)
                native._register_attribute(name, native_child._type(), native_child)
        return native

    return torch.jit._recursive.wrap_cpp_module(convert(model))
