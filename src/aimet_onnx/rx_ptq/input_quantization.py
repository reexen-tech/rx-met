"""Consumer input grids where a tensor-wide quantizer cannot model Torch I/O."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import onnx

from aimet_common import libpymo, libquant_info
from aimet_common.quantsim_config.compute_ops import is_compute_op, is_grid_preserving_op
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import save_model_with_external_weights

from .linear import linear_groups


def input_quantizer(sim, op, index):
    """Resolve the grid actually used at an original ONNX input port."""
    roles = getattr(sim, "_input_roles", None)
    if roles is not None and roles.is_control_port(op.name, index):
        return None
    alias = getattr(sim, "_rx_input_quantizers", {}).get((op.name, index))
    if alias is not None:
        return sim.qc_quantize_op_dict[alias]
    names = roles.input_names(op.name) if roles is not None else None
    if names is None:
        names = [product.name for product in op.inputs]
    if index >= len(names) or not names[index]:
        return None
    name = names[index]
    group = getattr(sim, '_rx_linear_groups', {}).get(op.name)
    if group is not None and index == 0:
        name = group['input']
    quantizer = sim.qc_quantize_op_dict.get(name)
    if quantizer is not None:
        return quantizer
    resolver = getattr(sim, "_get_enabled_quantizer", None)
    if resolver is not None and name in sim.connected_graph.get_all_products():
        return resolver(name)
    return None


class InputQuantizationSim(QuantizationSimModel):
    """Adapt consumer input grids without changing the clean graph.

    Generic QuantSim propagates input grids upstream through data movement.
    Native recurrent outputs already own their quantization, and Concat has a
    separate shared input grid in Torch. These boundaries require port-local
    QDQ. MatMul inputs also need independent grids to preserve producer output
    precision. Shared graph inputs use consumer grids instead of a common QDQ.
    """

    def __init__(self, model, **kwargs):
        self._rx_original_onnx = copy.deepcopy(model)
        self._rx_linear_groups = linear_groups(model)
        self._rx_input_quantizers = {}
        self._rx_input_owned_io = set()
        super().__init__(model, **kwargs)

    def _prepare_runtime_model(self):
        super()._prepare_runtime_model()
        original_nodes = list(self._rx_original_onnx.graph.node)
        producers = {name: node for node in original_nodes for name in node.output}
        runtime_nodes = {node.name: node for node in self.model.model.graph.node}
        occupied = set(self.connected_graph.get_all_products()) | set(runtime_nodes)
        native_outputs = getattr(self, "_rx_native_outputs", set())
        graph_inputs = {value.name for value in self._rx_original_onnx.graph.input}

        def origin(name):
            while name in producers and is_grid_preserving_op(producers[name].op_type):
                name = producers[name].input[0]
            return name

        matrix_nodes = {node.name for node in original_nodes
                        if node.op_type == "MatMul" or node.name in self._rx_linear_groups}
        consumers = {}
        for node in original_nodes:
            if is_grid_preserving_op(node.op_type):
                continue
            for index, name in enumerate(node.input):
                if name and not self._input_roles.is_control_port(node.name, index):
                    consumers.setdefault(origin(name), set()).add((node.name, index))
        local_inputs = {origin(name) for node in original_nodes
                        for name in node.input if self._is_quantizable_dtype(name)
                        and (node.op_type == "Concat"
                             or len(consumers.get(origin(name), ())) > 1)} & graph_inputs
        native_ports = {name for adapter in getattr(self, "_rx_native_recurrent", {}).values()
                        for name in (adapter.inputs[0], *adapter.inputs[5:]) if name}
        native_sources = {origin(name) for name in native_ports}
        native_inputs = native_sources & graph_inputs
        local_inputs.update(native_inputs)
        # Only the native input path bypasses shared grids, not other producers.
        native_carriers = set(native_inputs)
        for name in native_ports:
            if origin(name) not in native_inputs:
                continue
            while name in producers and is_grid_preserving_op(producers[name].op_type):
                native_carriers.add(name)
                name = producers[name].input[0]
        exposed_inputs = (local_inputs | native_carriers) & {
            value.name for value in self._rx_original_onnx.graph.output
        }
        self._rx_input_owned_io.update(native_carriers - exposed_inputs)
        # An input returned by the model keeps its output QDQ. Other branches
        # read the original input before applying their own port grids.
        raw_inputs = {name + "_updated": name for name in exposed_inputs}
        for runtime in runtime_nodes.values():
            if runtime.op_type != "QcQuantizeOp":
                for index, name in enumerate(runtime.input):
                    if name in raw_inputs:
                        runtime.input[index] = raw_inputs[name]
        pending = []
        for node in original_nodes:
            if not is_compute_op(node.op_type) or node.name in getattr(self, "_rx_native_recurrent", {}):
                continue
            shared = None
            for index, name in enumerate(node.input):
                if self._input_roles.is_control_port(node.name, index):
                    continue
                if not name or name in self.param_names or not self._is_quantizable_dtype(name):
                    continue
                source = origin(name)
                own_grid = self.qc_quantize_op_dict.get(name)
                matrix_input = node.name in matrix_nodes and (source in producers or (
                    name in producers and own_grid is not None and own_grid.enabled))
                if (source not in native_outputs | local_inputs | native_sources and node.op_type != "Concat"
                        and not matrix_input):
                    continue
                template = self._get_enabled_quantizer(name)
                if template is None and source in native_sources:
                    # Native ownership can disable the propagated input grid.
                    template = self.qc_quantize_op_dict.get(name) or self.qc_quantize_op_dict.get(source)
                if template is None:
                    continue  # Preserve JSON1 exclusions and disabled input grids.
                identity = hashlib.sha256(node.name.encode("utf-8")).hexdigest()[:16]
                alias = f"__rx_input_{identity}_{index}"
                if any(name in occupied for name in (alias, alias + "_updated", "QcQuantizeOp_" + alias)):
                    raise ValueError(f"Reserved runtime tensor name is already used: {alias}")
                quantizer = shared if node.op_type == "Concat" else None
                if quantizer is None:
                    quantizer = QcQuantizeOp(
                        quant_info=libquant_info.QcQuantizeInfo(),
                        quant_scheme=template.quant_scheme,
                        op_mode=libpymo.TensorQuantizerOpMode.updateStats,
                        bitwidth=template.bitwidth,
                    )
                    quantizer.data_type = template.data_type
                    quantizer.use_symmetric_encodings = template.use_symmetric_encodings
                    quantizer.use_strict_symmetric = template.use_strict_symmetric
                    quantizer.use_unsigned_symmetric = template.use_unsigned_symmetric
                    if node.op_type == "Concat":
                        shared = quantizer
                self._rx_input_quantizers[node.name, index] = alias
                if source in local_inputs and source not in exposed_inputs:
                    self._rx_input_owned_io.add(source)
                self.qc_quantize_op_dict[alias] = quantizer
                self.activation_names.append(alias)
                runtime = runtime_nodes[node.name]
                pending.extend([
                    onnx.helper.make_node("Identity", [runtime.input[index]], [alias], name=alias),
                    onnx.helper.make_node(
                        "QcQuantizeOp", [alias], [alias + "_updated"],
                        name="QcQuantizeOp_" + alias, domain=self._op_domain,
                        op_name=alias, quant_info=libpymo.PtrToInt64(quantizer.quant_info),
                    ),
                ])
                runtime.input[index] = alias + "_updated"
        # Only the native operator owns its output. Quantize at downstream
        # compute inputs, after any frame selection, rather than before branches.
        for name in native_outputs | self._rx_input_owned_io:
            if name in self.qc_quantize_op_dict:
                self.qc_quantize_op_dict[name].enabled = False
        if native_outputs:
            self._rx_native_io.update(native_outputs)
        self.model.model.graph.node.extend(pending)
        self._producers = {name: node for node in self.model.model.graph.node for name in node.output}

    def export(self, path, filename_prefix, export_model=True):
        super().export(path, filename_prefix, export_model=False)
        if export_model:
            model = copy.deepcopy(self._rx_original_onnx)
            onnx.external_data_helper.convert_model_from_external_data(model)
            output = str(Path(path) / f"{filename_prefix}.onnx")
            if model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
                save_model_with_external_weights(model, output, location=f"{filename_prefix}.data",
                                                all_tensors_to_one_file=True)
            else:
                onnx.save_model(model, output)
