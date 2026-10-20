"""Dataset inference and prepared-label loading for ONNX PTQ evaluation."""

import json
from pathlib import Path

from aimet_common.evaluation import classification_accuracy, compare_output_batches, compare_outputs
from .calibration import inspect_model_input_dtypes, inspect_model_inputs, iter_calibration_feeds


def load_onnx_validation_inputs(model, directory, *, labels=None):
    directory = Path(directory)
    feeds = list(iter_calibration_feeds(directory, inspect_model_inputs(model),
        input_dtypes=inspect_model_input_dtypes(model)))
    manifest_path = directory.parent / 'manifest.json'
    if labels is None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if 'val_files' in manifest:
            if 'val_labels' not in manifest:
                raise ValueError('Validation labels are missing; rerun prepare_onnx_ptq_data.py')
            files, targets = manifest['val_files'], manifest['val_labels']
            if len(files) != len(targets) or len(set(files)) != len(files):
                raise ValueError('Validation manifest file/label counts are inconsistent')
            actual = sorted(directory.glob('*.npz')) or sorted(directory.glob('*.npy'))
            if {path.name for path in actual} != set(files):
                raise ValueError('Validation files differ from the prepared manifest')
            by_name = dict(zip(files, targets))
            labels = [by_name[path.name] for path in actual]
    return feeds, labels


def evaluate_onnx_dataset(source, feeds, *, reference=None, labels=None,
                          providers=('CPUExecutionProvider',)):
    """Reuse one session for every feed, returning outputs and dataset metrics."""
    if hasattr(source, 'session'):
        session = source.session
    elif hasattr(source, 'run'):
        session = source
    else:
        import onnxruntime as ort
        session = ort.InferenceSession(str(source), providers=list(providers))
    outputs = []
    for feed in feeds:
        values = session.run(None, feed)
        compare_outputs(values, values)
        outputs.append(values)
    if not outputs:
        raise ValueError('Evaluation dataset is empty')
    report = {'feed_count': len(outputs)}
    if reference is not None:
        report['comparison'] = compare_output_batches(reference, outputs)
    if labels is not None:
        report.update(classification_accuracy(outputs, labels))
    return outputs, report
