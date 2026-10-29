"""Small compiler-encoding documents shared by normalization regressions."""
import copy

from aimet_common import compiler_encodings as shared


def encoding(**updates):
    result = dict(bitwidth=8, dtype='int', is_symmetric='True', enc_type='PER_TENSOR', scale=0.125,
                  offset=0, min=-16.0, max=15.875)
    result.update(updates)
    return result


def document(entry):
    return {'activation_encodings': {'conv': {'input': [entry]}},
            'param_encodings': {'conv.weight': [copy.deepcopy(entry)]}}
