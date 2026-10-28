"""Native CPU regressions for the lifetime of ConnectedGraph analysis traces."""

import gc
import unittest
import weakref
from unittest.mock import patch

import torch
from torch import nn

from aimet_torch.meta import connectedgraph as cg_module
from aimet_torch.meta._trace import trace_for_connected_graph
from aimet_torch.meta.connectedgraph import ConnectedGraph


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, value):
        return self.relu(self.linear(value))


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.alias = self.shared
        self.relu = nn.ReLU()

    def forward(self, value):
        return self.relu(self.shared(value)) + self.relu(self.alias(value))


class ExportedModel(LinearModel):
    @torch.jit.export
    def extra(self, value):
        return self.linear(value)


class ChildModel(nn.Module):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def forward(self, value):
        return self.child(value)


class OptionalStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.register_parameter("optional_parameter", None)
        self.register_buffer("optional_buffer", None)
        self.add_module("optional_child", None)

    def forward(self, value):
        return self.linear(value)


class NativeObjectModel(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.quantize_per_tensor(torch.randn(4, 4), 0.05, 0, torch.qint8)
        self.packed = torch.ops.quantized.linear_prepack(weight, torch.randn(4))

    def forward(self, value):
        return torch.ops.quantized.linear_dynamic(value, self.packed)


class RecurrentModel(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.recurrent = cell(4, 4, batch_first=True)
        self.linear = nn.Linear(4, 2)

    def forward(self, value):
        output, state = self.recurrent(value)
        return self.linear(output), state


class ContainerModel(nn.Module):
    def forward(self, values):
        return {"sum": values[0] + values[1], "first": values[0]}


class MappingModel(nn.Module):
    def forward(self, values):
        return [values["left"] + values["right"], values["left"]]


class FailingModel(LinearModel):
    def forward(self, value):
        self.linear(value)
        raise RuntimeError("expected tracing failure")


def _global_trace_counts():
    return (
        len(torch.jit._state._python_cu.get_functions()),
        len(torch.jit._recursive.concrete_type_store.methods_compiled),
    )


def _structure(value):
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(_structure(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _structure(item)) for key, item in value.items())
    return repr(value)


def _signature(graph):
    return (
        tuple(
            (
                op.name, op.dotted_name, op.type, op.get_module(),
                tuple(product.name for product in op.inputs),
                tuple(product.name for product in op.outputs),
            )
            for op in graph.ordered_ops
        ),
        tuple(
            (
                name, _structure(product.shape), product.is_parm,
                product.is_model_input, product.is_const,
                product.producer.name if product.producer else None,
                tuple(consumer.name for consumer in product.consumers),
            )
            for name, product in sorted(graph.get_all_products().items())
        ),
        _structure(graph.input_structure),
        _structure(graph.output_structure),
    )


class ConnectedGraphLifetimeTest(unittest.TestCase):
    def test_topology_parameters_and_recurrent_outputs_match_existing_trace(self):
        cases = [
            (LinearModel(), torch.randn(2, 4)),
            (SharedModel(), torch.randn(2, 4)),
            (ExportedModel(), torch.randn(2, 4)),
            (ChildModel(ExportedModel()), torch.randn(2, 4)),
            (OptionalStateModel(), torch.randn(2, 4)),
            (RecurrentModel(nn.GRU), torch.randn(2, 3, 4)),
            (RecurrentModel(nn.LSTM), torch.randn(2, 3, 4)),
        ]
        for model, inputs in cases:
            with self.subTest(model=type(model).__name__):
                with patch.object(cg_module, "trace_for_connected_graph", torch.jit.trace):
                    expected = ConnectedGraph(model, inputs)
                actual = ConnectedGraph(model, inputs)
                self.assertEqual(_signature(actual), _signature(expected))
                self.assertTrue(any(p.is_parm for p in actual.get_all_products().values()))

    def test_disposable_graphs_do_not_accumulate_global_types_or_functions(self):
        inputs = torch.randn(2, 4)
        live_original = LinearModel().eval()
        live_trace = torch.jit.trace(live_original, inputs, check_trace=False)
        live_script = torch.jit.script(nn.ReLU())
        counts = _global_trace_counts()
        for _ in range(12):
            model = SharedModel().eval()
            graph = ConnectedGraph(model, inputs)
            reference = weakref.ref(model)
            self.assertIsNotNone(graph.get_op_from_module_name("SharedModel.shared"))
            del graph, model
            gc.collect()
            self.assertIsNone(reference())
            self.assertEqual(_global_trace_counts(), counts)
            torch.testing.assert_close(live_trace(inputs), live_original(inputs))
            torch.testing.assert_close(live_script(inputs), inputs.relu())

        for _ in range(3):
            graph = ConnectedGraph(ExportedModel().eval(), inputs)
            del graph
            gc.collect()
            self.assertEqual(_global_trace_counts(), counts)

    def test_trace_options_containers_and_exception_cleanup(self):
        inputs = torch.randn(2, 4)
        counts = _global_trace_counts()
        for model, values in (
            (ContainerModel(), (inputs, inputs + 1)),
            (ContainerModel(), [inputs, inputs + 1]),
            (MappingModel(), {"left": inputs, "right": inputs + 1}),
        ):
            traced = trace_for_connected_graph(
                model, (values,), check_trace=True, strict=False
            )
            torch.testing.assert_close(traced(values), model(values))
            del traced
        previous = torch.jit._trace._trace_module_map
        with self.assertRaisesRegex(RuntimeError, "expected tracing failure"):
            trace_for_connected_graph(FailingModel(), inputs, check_trace=False)
        self.assertIs(torch.jit._trace._trace_module_map, previous)
        gc.collect()
        self.assertEqual(_global_trace_counts(), counts)

    def test_existing_script_modules_keep_their_owners(self):
        inputs = torch.randn(2, 4)
        scripted = torch.jit.script(LinearModel().eval())
        expected = scripted(inputs)
        counts = _global_trace_counts()
        self.assertIs(
            trace_for_connected_graph(scripted, inputs, check_trace=False), scripted
        )
        for _ in range(3):
            parent = ChildModel(scripted)
            traced = trace_for_connected_graph(parent, inputs, check_trace=False)
            torch.testing.assert_close(traced(inputs), expected)
            del traced, parent
            gc.collect()
            self.assertEqual(_global_trace_counts(), counts)
            torch.testing.assert_close(scripted(inputs), expected)

    def test_native_script_object_attributes_remain_available(self):
        if torch.backends.quantized.engine == "none":
            self.skipTest("A CPU quantized backend is required for a native ScriptObject")
        inputs = torch.randn(2, 4)
        model = NativeObjectModel()
        self.assertTrue(torch._C._jit_is_script_object(model.packed))
        expected = model(inputs)
        counts = _global_trace_counts()
        for _ in range(3):
            traced = trace_for_connected_graph(model, inputs, check_trace=False)
            torch.testing.assert_close(traced(inputs), expected)
            del traced
            gc.collect()
            self.assertEqual(_global_trace_counts(), counts)

    def test_custom_module_factories_cannot_bypass_local_ownership(self):
        inputs = torch.randn(2, 4)
        model = LinearModel().eval()
        counts = _global_trace_counts()
        traced = trace_for_connected_graph(
            model, inputs, check_trace=False, _module_class=None
        )
        torch.testing.assert_close(traced(inputs), model(inputs))
        with self.assertRaisesRegex(ValueError, "custom _module_class"):
            trace_for_connected_graph(
                model, inputs, check_trace=False, _module_class=lambda *args: None
            )
        del traced
        gc.collect()
        self.assertEqual(_global_trace_counts(), counts)


if __name__ == "__main__":
    unittest.main()
