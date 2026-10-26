"""Resolve recorded KWS artifacts, including reports moved out of a container.

Compiler artifacts are selected by the exporter return paths in comparison.json.
Historical .encodings and current .json are both supported for both frontends; filenames
alone cannot distinguish a final compiler document from raw AIMET encodings.
"""

import json
from pathlib import Path


def _recorded_path(directory, recorded, stage):
    if not isinstance(recorded, str) or not recorded:
        raise ValueError(f"Missing recorded {stage} artifact path")
    path = Path(recorded)
    if '..' in path.parts:
        raise ValueError(f"Artifact path must not contain '..': {recorded}")
    candidates = []
    # Preserve the full stage/run/exports suffix when host and container roots differ.
    if stage in path.parts:
        index = path.parts.index(stage)
        candidates.append(directory.joinpath(*path.parts[index:]))
    candidates.append(path if path.is_absolute() else directory / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Recorded artifact does not exist: {recorded}")


def resolve_artifact_paths(directory, report=None):
    """Use explicit report paths, never a glob over .json/.encodings files.

    Older reports omitted the metadata path. Only that path uses the ONNX PTQ
    metadata naming contract, derived from the recorded ONNX model, for compatibility.
    """
    directory = Path(directory).resolve()
    if report is None:
        report = json.loads((directory / 'comparison.json').read_text(encoding='utf-8'))
    artifacts = report.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError('comparison.json must contain explicit artifact paths')
    result = {}
    for key in ('torch_onnx', 'torch_encodings', 'onnx_onnx', 'onnx_encodings',
                'onnx_metadata', 'onnx_aimet_encodings'):
        if key not in artifacts and key in ('onnx_metadata', 'onnx_aimet_encodings'):
            continue
        stage = 'torch_ptq' if key.startswith('torch_') else 'onnx_ptq'
        result[key] = _recorded_path(directory, artifacts.get(key), stage)
    if 'onnx_metadata' not in result:
        metadata = result['onnx_onnx'].with_suffix('.meta.json')
        if not metadata.is_file():
            raise FileNotFoundError(f'Historical ONNX PTQ metadata does not exist: {metadata}')
        result['onnx_metadata'] = metadata
    return result


def load_final_compiler_encodings(path):
    """Identify final schema 4 documents without normalizing historical defects.

    Numeric acceptance remains the independent strict validator's responsibility;
    audits must still be able to inspect malformed grids in old final documents.
    """
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(document, dict) or type(document.get('schema_version')) is not int
            or document['schema_version'] != 4
            or not isinstance(document.get('activation_encodings'), dict)
            or not isinstance(document.get('param_encodings'), dict)):
        raise ValueError(f'Expected a final compiler schema 4 document, not raw AIMET encodings: {path}')
    return document
