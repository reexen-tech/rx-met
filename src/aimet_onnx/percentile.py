"""ONNX tensor access and native-grid adapter for shared percentile calibration."""

from contextlib import contextmanager
import copy

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from aimet_common import libpymo
from aimet_common.defs import QuantizationDataType, QuantScheme
from aimet_common.percentile import PercentileCalibrator
from aimet_onnx.utils import OrtInferenceSession, add_hook_to_get_activation


def uses_shared_percentile(quantizer):
    return (quantizer.quant_scheme == QuantScheme.post_training_percentile
            and quantizer.data_type == QuantizationDataType.int)


def update_stats(quantizer, array):
    if quantizer.is_encoding_frozen() or not quantizer.enabled:
        return
    array = np.asarray(array, dtype=np.float32)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, copy=True, order="C")
    tensor = torch.from_numpy(array).to(getattr(quantizer, "_rx_calibration_device", "cpu"))
    shape = tuple(quantizer._tensor_quantizer.getShape())
    if quantizer.quant_info.blockSize:
        axis = quantizer.quant_info.blockAxis
        block_size = quantizer.quant_info.blockSize
        tensor = tensor.reshape(*tensor.shape[:axis], tensor.shape[axis] // block_size,
                                block_size, *tensor.shape[axis + 1:])
        shape = (*shape[:axis + 1], 1, *shape[axis + 1:])
    if quantizer._percentile_calibrator is None:
        quantizer._percentile_calibrator = PercentileCalibrator(
            shape, percentile=quantizer._percentile_value)
    quantizer._percentile_calibrator.update_stats(tensor)


def compute_encodings(quantizer):
    calibrator = quantizer._percentile_calibrator
    if calibrator is None:
        raise RuntimeError("Percentile quantizer has no shared calibration observations")
    # Frontends share the statistics/range calculation; TfEncoding only adapts
    # the resulting grid to the native runtime's additive-offset convention.
    num_steps = (1 << quantizer.bitwidth) - 1
    symmetric = quantizer.use_symmetric_encodings
    signed_steps = num_steps - int(symmetric and quantizer.use_strict_symmetric)
    minimum, maximum = calibrator.compute_encodings(signed_steps, symmetric)
    unsigned = None
    if symmetric and quantizer.use_unsigned_symmetric:
        unsigned = calibrator.compute_encodings(num_steps, False)
    encodings = []
    for index, (low, high) in enumerate(zip(minimum.flatten(), maximum.flatten())):
        signed_grid = symmetric
        steps = signed_steps
        if unsigned is not None and unsigned[0].flatten()[index] >= 0:
            low, high = (bound.flatten()[index] for bound in unsigned)
            signed_grid = False
            steps = num_steps
        if signed_grid:
            offset = -((steps + 1) // 2)
        else:
            offset = None
        # The shared analyzer has already aligned symmetric bounds. Recover
        # scale from that range, as Torch does, without a second float32 rounding.
        delta = ((high - low) / steps).item()
        if offset is None:
            offset = round(low.item() / delta)
        encoding = libpymo.TfEncoding()
        encoding.bw = quantizer.bitwidth
        encoding.delta = delta
        encoding.offset = offset
        encoding.min = offset * delta
        encoding.max = (offset + steps) * delta
        encodings.append(encoding)
    return encodings


class _CalibrationSession:
    def __init__(self, original, collector, quantizers, inputs):
        self._original = original
        self._collector = collector
        self._quantizers = quantizers
        self._inputs = inputs

    def __getattr__(self, name):
        return getattr(self._original, name)

    def run(self, output_names, input_feed, run_options=None):
        requested = (list(output_names) if output_names else
                     [value.name for value in self._original.get_outputs()])
        # Do not fetch graph inputs as outputs: ORT >= 1.19 can corrupt these
        # buffers. Observe them directly from the caller's feed instead.
        fetched = list(dict.fromkeys(requested + [
            name for name in self._quantizers if name not in self._inputs]))
        runtime_names = [name for name in fetched if name not in self._inputs]
        outputs = self._collector.run(runtime_names, input_feed, run_options)
        values = dict(zip(runtime_names, outputs))
        values.update({name: input_feed[name] for name in fetched if name in self._inputs})
        for name, quantizer in self._quantizers.items():
            quantizer.update_encoding_stats(input_feed[name] if name in self._inputs else values[name])
        return [values[name] for name in requested]

    def run_with_iobinding(self, *args, **kwargs):
        raise NotImplementedError("Percentile calibration requires session.run with NumPy feeds")

    run_with_ort_values = run_with_iobinding
    run_async = run_with_iobinding


@contextmanager
def calibration_context(sim, quantizers):
    selected = {name: q for name, q in quantizers.items()
                if uses_shared_percentile(q) and not q.is_encoding_frozen()}
    if not selected:
        yield
        return
    original = sim.session
    modes = {name: q.op_mode for name, q in selected.items()}
    device = "cpu"
    for provider in sim.providers:
        name, options = provider if isinstance(provider, tuple) else (provider, {})
        if name == "CUDAExecutionProvider":
            device = f"cuda:{options.get('device_id', 0)}"
            break
    initializers = {value.name: value for value in sim.model.model.graph.initializer}
    activations = {}
    try:
        for name, quantizer in selected.items():
            # Keep histogram/range arithmetic on the same device as Torch.
            # CPU and CUDA scalar division can differ by one float32 ULP.
            quantizer._rx_calibration_device = device
            if name in sim.param_names:
                if name not in initializers:
                    raise RuntimeError(f"Percentile parameter has no constant initializer: {name}")
                quantizer.update_encoding_stats(numpy_helper.to_array(initializers[name]))
                quantizer.compute_encodings()
                if getattr(quantizer, "_rx_calibration_passthrough", False):
                    quantizer.op_mode = libpymo.TensorQuantizerOpMode.passThrough
            else:
                quantizer.op_mode = libpymo.TensorQuantizerOpMode.passThrough
                activations[name] = quantizer
        if activations:
            model = copy.deepcopy(sim.model.model)
            inputs = {value.name for value in model.graph.input}
            outputs = {value.name for value in model.graph.output}
            for name in activations.keys() - inputs - outputs:
                add_hook_to_get_activation(model, name)
            collector = OrtInferenceSession(model, sim.providers, sim._ort_session_options)
            sim.session = _CalibrationSession(original, collector, activations, inputs)
        yield
    except BaseException:
        for name, mode in modes.items():
            selected[name].op_mode = mode
        raise
    finally:
        sim.session = original
