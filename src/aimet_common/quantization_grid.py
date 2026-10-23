"""Integer grids using real = scale * (q - zero_point), independent of symmetry."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


def integer_bounds(bitwidth: int, signed: bool, *, strict: bool = False):
    if type(bitwidth) is not int or not 1 <= bitwidth <= 64:
        raise ValueError('bitwidth must be an integer between 1 and 64')
    if type(signed) is not bool:
        raise ValueError('signed must be a boolean')
    low = -(1 << (bitwidth - 1)) if signed else 0
    high = (1 << (bitwidth - int(signed))) - 1
    return low + int(strict and signed), high


def symmetric_offset(qmin: int, qmax: int):
    return 0 if qmin >= 0 else -round((qmin + qmax) / 2)


@dataclass(frozen=True)
class IntegerGrid:
    scale: Any
    zero_point: Any
    bitwidth: int
    signed: bool
    qmin: int | None = None
    qmax: int | None = None

    def __post_init__(self):
        low, high = integer_bounds(self.bitwidth, self.signed)
        qmin = low if self.qmin is None else self.qmin
        qmax = high if self.qmax is None else self.qmax
        if type(qmin) is not int or type(qmax) is not int or not low <= qmin < qmax <= high:
            raise ValueError('integer range does not fit dtype')
        scale = np.asarray(self.scale, dtype=np.float64)
        point = np.asarray(self.zero_point, dtype=object)
        if not scale.size or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('scale must be positive and finite')
        # Comparing INT64 bounds in float64 rounds qmax up to the first invalid
        # code. Keep the integer comparison exact before any numeric conversion.
        for value in point.flat:
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or (not isinstance(value, Integral)
                        and (not np.isfinite(value) or value != int(value)))
                    or not low <= int(value) <= high):
                raise ValueError('zero_point must be integral and fit dtype range')
        if point.ndim and point.shape != scale.shape:
            raise ValueError('zero_point shape differs from scale')
        object.__setattr__(self, 'qmin', qmin)
        object.__setattr__(self, 'qmax', qmax)

    @property
    def dtype(self):
        return f'{"INT" if self.signed else "UINT"}{self.bitwidth}'

    @property
    def real_min(self):
        return self._real_bound(self.qmin)

    @property
    def real_max(self):
        return self._real_bound(self.qmax)

    def _real_bound(self, bound):
        points = np.asarray(self.zero_point, dtype=object)
        codes = np.asarray([bound - int(point) for point in points.flat], dtype=np.float64)
        return np.asarray(self.scale, dtype=np.float64) * codes.reshape(points.shape)

    def additive_offset(self, *, index_domain=False):
        points = np.asarray(self.zero_point, dtype=object if self.bitwidth == 64 else np.int64)
        return (self.qmin if index_domain else 0) - points

    @classmethod
    def from_offset(cls, scale, offset, bitwidth, signed, *, index_domain=False,
                    qmin=None, qmax=None):
        low, high = integer_bounds(bitwidth, signed)
        low = low if qmin is None else qmin
        high = high if qmax is None else qmax
        values = np.asarray(offset, dtype=np.float64)
        if not np.isfinite(values).all() or (values != np.rint(values)).any():
            raise ValueError('offset must be finite and integral')
        values = np.asarray(offset, dtype=object if bitwidth == 64 else np.int64)
        point = (low if index_domain else 0) - values
        return cls(scale, point.tolist(), bitwidth, signed, low, high)

    @classmethod
    def from_compiler_record(cls, record):
        bits = record['bitwidth']
        dtype = record['dtype']
        if dtype not in (f'INT{bits}', f'UINT{bits}'):
            raise ValueError(f'unsupported integer dtype: {dtype!r}')
        return cls(record['scale'], record['zero_point'], bits, dtype.startswith('INT'),
                   record.get('qmin'), record.get('qmax'))

    @classmethod
    def from_aimet_legacy_records(cls, records, *, signed):
        if not records:
            raise ValueError('empty AIMET encoding records')
        first = records[0]
        signed = first.get('signed', signed)
        low, high = integer_bounds(first['bitwidth'], signed)
        low, high = first.get('qmin', low), first.get('qmax', high)
        for entry in records:
            if any(entry.get(key) != first.get(key) for key in
                   ('bitwidth', 'signed', 'qmin', 'qmax', 'offset_domain')):
                raise ValueError('AIMET channel records disagree on integer domain')
        scales = [entry['scale'] for entry in records]
        offsets = [entry['offset'] for entry in records]
        minimum = np.asarray([entry['min'] for entry in records])
        maximum = np.asarray([entry['max'] for entry in records])
        domain = first.get('offset_domain')
        if domain not in (None, 'unsigned_index', 'target'):
            raise ValueError(f'unsupported offset domain: {domain!r}')
        candidates = []
        for index_domain in ((True, False) if domain is None else (domain == 'unsigned_index',)):
            try:
                grid = cls.from_offset(scales, offsets, first['bitwidth'], signed,
                                       index_domain=index_domain, qmin=low, qmax=high)
            except ValueError:
                continue
            tolerance = np.asarray(scales) * 1e-5
            if (np.all(np.abs(grid.real_min-minimum) <= tolerance+np.abs(minimum)*1e-6)
                    and np.all(np.abs(grid.real_max-maximum) <= tolerance+np.abs(maximum)*1e-6)):
                candidates.append(grid)
        if not candidates:
            raise ValueError('AIMET offset/scale disagree with saved real range')
        if len(candidates) > 1 and candidates[0].zero_point != candidates[1].zero_point:
            raise ValueError('AIMET offset domain is ambiguous')
        return candidates[0]
