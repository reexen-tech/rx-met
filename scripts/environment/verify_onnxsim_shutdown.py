"""Exercise onnxsim after Torch; the caller checks stderr after normal exit."""


def main():
    import torch
    import onnxsim
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original_options = ort.SessionOptions

    def single_thread_options():
        options = original_options()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return options

    # onnxsim's internal ORT sessions also need bounded thread pools in this probe.
    ort.SessionOptions = single_thread_options
    try:
        model = helper.make_model(
            helper.make_graph([
                helper.make_node('Add', ['a', 'b'], ['constant_sum']),
                helper.make_node('Add', ['x', 'constant_sum'], ['y']),
            ], 'shutdown-regression',
                [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2])],
                [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2])], [
                    numpy_helper.from_array(np.array([1, 2], dtype=np.float32), 'a'),
                    numpy_helper.from_array(np.array([3, 4], dtype=np.float32), 'b'),
                ]),
            opset_imports=[helper.make_opsetid('', 17)], ir_version=9,
        )
        for _ in range(3):
            result, ok = onnxsim.simplify(model, check_n=0)
            assert ok
            onnx.checker.check_model(result)
            assert len(result.graph.node) == 1, 'constant Add was not folded'
            session = ort.InferenceSession(result.SerializeToString(),
                                           sess_options=single_thread_options(),
                                           providers=['CPUExecutionProvider'])
            actual = session.run(None, {'x': np.array([5, 6], dtype=np.float32)})[0]
            np.testing.assert_array_equal(actual, [9, 12])
        print('onnxsim constant folding and ORT CPU outputs verified')
    finally:
        ort.SessionOptions = original_options


if __name__ == '__main__':
    main()
