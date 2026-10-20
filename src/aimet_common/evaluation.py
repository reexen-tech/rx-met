"""Numeric output comparisons and classification metrics shared by frontends."""

import numpy as np


def compare_outputs(reference, actual):
    if len(reference) != len(actual) or not reference:
        raise RuntimeError('Output counts differ or no outputs were provided')
    report = []
    for index, (ref, value) in enumerate(zip(reference, actual)):
        ref, value = np.asarray(ref), np.asarray(value)
        if ref.shape != value.shape:
            raise RuntimeError(f'Output {index} shape differs: {ref.shape} vs {value.shape}')
        if not ref.size:
            raise ValueError(f'Output {index} is empty')
        if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(value)):
            raise ValueError(f'Output {index} contains NaN or Inf')
        diff = value.astype(np.float64) - ref.astype(np.float64)
        report.append({
            'index': index, 'shape': list(ref.shape),
            'max_abs': float(np.max(np.abs(diff))),
            'mean_abs': float(np.mean(np.abs(diff))),
            'rmse': float(np.sqrt(np.mean(np.square(diff)))),
            'reference_max_abs': float(np.max(np.abs(ref))),
        })
    return report


def compare_output_batches(reference, actual):
    """Aggregate errors by element count, without averaging per-batch RMSEs."""
    if len(reference) != len(actual) or not reference:
        raise RuntimeError('Evaluation batch counts differ or the dataset is empty')
    totals = None
    for expected, values in zip(reference, actual):
        report = compare_outputs(expected, values)
        if totals is None:
            totals = [dict(item, value_count=0, absolute_sum=0., squared_sum=0.) for item in report]
        if len(report) != len(totals):
            raise RuntimeError('Output count changed between evaluation batches')
        for total, item, array in zip(totals, report, expected):
            count = np.asarray(array).size
            total['value_count'] += count
            total['absolute_sum'] += item['mean_abs'] * count
            total['squared_sum'] += item['rmse'] ** 2 * count
            total['max_abs'] = max(total['max_abs'], item['max_abs'])
            total['reference_max_abs'] = max(total['reference_max_abs'], item['reference_max_abs'])
    for total in totals:
        total['mean_abs'] = total.pop('absolute_sum') / total['value_count']
        total['rmse'] = float(np.sqrt(total.pop('squared_sum') / total['value_count']))
    return totals


def classification_accuracy(outputs, labels, *, output_index=0):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu':
        raise ValueError('Classification labels must be a one-dimensional integer array')
    scores = [np.asarray(batch[output_index]) for batch in outputs]
    if not scores or any(value.ndim != 2 or not value.shape[1] for value in scores):
        raise ValueError('Classification outputs must have shape [batch, classes]')
    if len({value.shape[1] for value in scores}) != 1:
        raise ValueError('Classification class count changed between batches')
    scores = np.concatenate(scores)
    if not scores.shape[0] or scores.shape[0] != len(labels):
        raise ValueError('Classification label count differs from the output sample count')
    if not np.all(np.isfinite(scores)):
        raise ValueError('Classification outputs contain NaN or Inf')
    if np.any(labels < 0) or np.any(labels >= scores.shape[1]):
        raise ValueError('Classification label exceeds the model output class range')
    correct = int(np.count_nonzero(scores.argmax(axis=1) == labels))
    return {'sample_count': len(labels), 'correct_count': correct, 'top1_accuracy': correct / len(labels)}
