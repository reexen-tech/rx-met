"""Real ONNX percentile quantizer construction for calibration regressions."""

from aimet_common import libquant_info
from aimet_common.defs import QuantScheme
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode, TensorQuantizerParams


def quantizer(*, percentile=99.99, symmetric=True, shape=None, channel=None):
    params = TensorQuantizerParams(shape, channel_axis=channel) if shape else None
    q = QcQuantizeOp(libquant_info.QcQuantizeInfo(),
                    quant_scheme=QuantScheme.post_training_percentile,
                    op_mode=OpMode.updateStats, use_symmetric_encodings=symmetric,
                    tensor_quantizer_params=params)
    q.use_strict_symmetric = False
    q.use_unsigned_symmetric = False
    if channel is not None:
        q.enable_per_channel_quantization()
    q.set_percentile_value(percentile)
    return q
