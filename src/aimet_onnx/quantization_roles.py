"""Consumer-specific control paths that must not acquire data QDQ grids."""

from collections import defaultdict
import copy
import hashlib


_TRANSPARENT_OPS = frozenset({"Identity", "Cast", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten"})


class QuantizationInputRoles:
    """Protect standard Clip bounds, including transparent shared branches.

    Numeric producer inputs keep their ordinary quantization. Transparent views used
    as bounds read the pre-QDQ source, without changing data consumers.
    """

    def __init__(self, model):
        self._inputs = {node.name: tuple(node.input) for node in model.graph.node}
        self._producers = {name: copy.deepcopy(node) for node in model.graph.node for name in node.output if name}
        self._bounds = {(node.name, index): name for node in model.graph.node
                        if node.op_type == "Clip" and not node.domain
                        for index, name in enumerate(node.input) if index in (1, 2) and name}
        self.control_ports = set(self._bounds)
        self.control_only = set()
        consumers = defaultdict(set)
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name:
                    consumers[name].add((node.name, index))
        outputs = {item.name for item in model.graph.output}
        pending = list(self._bounds.values())
        while pending:
            name = pending.pop()
            if name in self.control_only or name in outputs or not consumers[name] <= self.control_ports:
                continue
            self.control_only.add(name)
            producer = self._producers.get(name)
            if producer is not None and not producer.domain and producer.op_type in _TRANSPARENT_OPS:
                port = (producer.name, 0)
                self.control_ports.add(port)
                pending.append(producer.input[0])

    def is_control_port(self, node_name, index):
        return (node_name, index) in self.control_ports

    def input_names(self, node_name):
        return self._inputs.get(node_name)

    def prepare_runtime_model(self, model):
        if not self._bounds:
            return
        runtime = {node.name: node for node in model.graph.node}
        occupied = {node.name for node in model.graph.node}
        occupied.update(name for node in model.graph.node for name in (*node.input, *node.output) if name)
        occupied.update(item.name for item in (*model.graph.input, *model.graph.output,
                                               *model.graph.initializer, *model.graph.value_info))
        raw_values, clones = {}, []

        def raw_value(name):
            if name in raw_values:
                return raw_values[name]
            producer = self._producers.get(name)
            if producer is None or producer.domain or producer.op_type not in _TRANSPARENT_OPS:
                return name
            source = raw_value(producer.input[0])
            original = runtime[producer.name]
            if name in self.control_only and original.input[0] == source:
                raw_values[name] = name
                return name
            identity = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
            output = f"__rx_control_{identity}"
            if output in occupied or output + "_view" in occupied:
                raise ValueError(f"Reserved control-path name is already used: {output}")
            clone = copy.deepcopy(producer)
            clone.name = output + "_view"
            clone.input[0] = source
            clone.output[0] = output
            clones.append(clone)
            occupied.update((output, clone.name))
            raw_values[name] = output
            return output

        for (node_name, index), source in self._bounds.items():
            runtime[node_name].input[index] = raw_value(source)
        model.graph.node.extend(clones)
