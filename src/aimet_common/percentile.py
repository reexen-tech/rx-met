# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""Shared histogram and percentile calibration for Torch and ONNX frontends.

Tensor arithmetic uses Torch on CPU or CUDA; frontend adapters own tensor access
and encoding storage. Importing aimet_common alone does not import Torch.
"""

import itertools
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from aimet_common.quantsim import _get_minimum_scale


@dataclass
class Histogram:
    histogram: torch.Tensor = None
    bin_edges: torch.Tensor = None
    min: Optional[torch.Tensor] = None
    max: Optional[torch.Tensor] = None


def _is_expandable(shape, target):
    return len(shape) <= len(target) and all(
        source in (1, dest) for source, dest in zip(reversed(shape), reversed(target))
    )


class HistogramObserver:
    """
    Observer for Histogram based calibration techniques (percentile, MSE)
    """

    def __init__(
        self, shape: tuple, num_bins: int, growth_limit: Optional[float] = None
    ):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.num_bins = num_bins
        self.stats = [Histogram() for _ in range(np.prod(self.shape, dtype=np.int32))]

        if growth_limit is None:
            growth_limit = float("inf")

        if growth_limit < 1.0:
            raise RuntimeError

        self.growth_limit = growth_limit
        self._histogram_range_limit = None

    # pylint: disable=too-many-locals
    @torch.no_grad()
    def collect_stats(self, input_tensor: torch.Tensor) -> List[Histogram]:
        if not _is_expandable(self.shape, input_tensor.shape):
            raise RuntimeError(
                f"Shape {self.shape} is incompatible with input of shape {input_tensor.shape}"
            )

        hist_stats = []
        input_shape = tuple(input_tensor.shape)
        histogram_shape = self.shape

        padded_histogram_shape = (
            *itertools.repeat(1, len(input_shape) - len(histogram_shape)),
            *histogram_shape,
        )

        for i, _ in enumerate(self.stats):
            hist_input = input_tensor

            for axis, dim in enumerate(padded_histogram_shape):
                if dim == 1:
                    continue
                # elements in current axis, ex: could be W*C, C, or 1 for input_shape [H, W, C]
                numel = np.prod(padded_histogram_shape[axis + 1 :], dtype=np.int32)
                # index where hist_input at current dimension will be sliced at
                index = (i // numel) % dim
                hist_input = torch.unsqueeze(
                    torch.select(hist_input, axis, index), axis
                )

            hist_min, hist_max = self._get_min_max(hist_input)

            bin_edges = self._create_bin_edges(
                min_val=hist_min, max_val=hist_max, device=input_tensor.device
            )
            histogram = self._count_bins(hist_input, bin_edges)

            hist_stats.append(Histogram(histogram, bin_edges, hist_min, hist_max))

        if not torch.stack([stat.min for stat in hist_stats]).isfinite().all():
            raise RuntimeError("A histogram needs at least one finite observation per channel")
        return hist_stats

    def _get_min_max(
        self, hist_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        isfinite = hist_input.isfinite()
        min = torch.where(isfinite, hist_input, float("inf")).min()
        max = torch.where(isfinite, hist_input, -float("inf")).max()
        return min, max

    def _create_bin_edges(self, min_val, max_val, device):
        # Narrow intervals far from zero can have several bins per float32 ULP.
        min_val, max_val = min_val.double(), max_val.double()
        # histc expands a constant by 0.5; large constants need a representable width per bin.
        half_width = (min_val.abs() * torch.finfo(torch.float64).eps * self.num_bins).clamp_min(0.5)
        min_val, max_val = (
            torch.where(min_val == max_val, min_val - half_width, min_val),
            torch.where(min_val == max_val, max_val + half_width, max_val),
        )

        step = (max_val - min_val) / self.num_bins
        edges = torch.arange(self.num_bins + 1, dtype=torch.float64, device=device) * step + min_val
        edges[-1] = max_val
        return edges

    def _count_bins(self, values, edges):
        # Interior edges belong to the bin on their right; saturate both tails.
        boundaries = edges[1:-1]
        if values.is_floating_point() and values.dtype != boundaries.dtype:
            # Round edges upwards so comparison in the input dtype is exact.
            rounded = boundaries.to(values.dtype)
            boundaries = torch.where(rounded.double() < boundaries,
                                     torch.nextafter(rounded, torch.full_like(rounded, torch.inf)), rounded)
        counts = torch.zeros(self.num_bins + 1, dtype=torch.int64, device=values.device)
        one = counts.new_ones(())
        for chunk in values.reshape(-1).split(1 << 20):
            indices = torch.bucketize(chunk.contiguous(), boundaries, right=True)
            # A spare bin ignores NaNs without compacting the input on the host.
            indices.masked_fill_(chunk.isnan(), self.num_bins)
            counts.scatter_add_(0, indices, one.expand_as(indices))
        return counts[:-1]

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-locals
    @torch.no_grad()
    def merge_stats(self, new_stats_list: List[Histogram], input_tensor: torch.Tensor):
        if self.stats[0].histogram is None:
            self.stats = new_stats_list

            # Compute histogram range limit based on the first input and the growth limit factor
            input0_min = torch.stack([stat.min for stat in new_stats_list]).double()
            input0_max = torch.stack([stat.max for stat in new_stats_list]).double()

            input0_range = input0_max - input0_min

            # A constant first observation must still produce finite growth limits.
            minimum_scale = _get_minimum_scale(self.num_bins)
            minimum_range = minimum_scale * self.num_bins

            zero_range_mask = (input0_range.abs() < minimum_range) | torch.isnan(input0_range) | torch.isinf(input0_range)
            if zero_range_mask.any():
                input0_min_safe = torch.where(zero_range_mask, torch.minimum(input0_min, torch.zeros_like(input0_min)), input0_min)
                input0_max_safe = torch.where(zero_range_mask, torch.maximum(input0_max, torch.zeros_like(input0_max)), input0_max)
                input0_range = torch.where(zero_range_mask,
                                          torch.full_like(input0_range, minimum_range),
                                          input0_range)
                input0_min = input0_min_safe
                input0_max = input0_max_safe

            left_limit = (input0_min - input0_range * self.growth_limit / 2).flatten()
            right_limit = (input0_max + input0_range * self.growth_limit / 2).flatten()
            self._histogram_range_limit = torch.stack((left_limit, right_limit), dim=1)
            return

        padded_shape = (1,) * (input_tensor.ndim - len(self.shape)) + self.shape
        channels = tuple(axis for axis, size in enumerate(padded_shape) if size > 1)
        reduced = tuple(axis for axis, size in enumerate(padded_shape) if size == 1)
        hist_inputs = input_tensor.permute(channels + reduced).reshape(len(new_stats_list), -1)
        for index, new_stats in enumerate(new_stats_list):
            curr_stats = self.stats[index]
            curr_input = hist_inputs[index]
            left_limit, right_limit = self._histogram_range_limit[index]

            updated_min = torch.minimum(new_stats.min.clamp(min=left_limit), curr_stats.bin_edges[0])
            updated_max = torch.maximum(new_stats.max.clamp(max=right_limit), curr_stats.bin_edges[-1])

            # if the current histogram can capture new_stats within in its range
            if updated_min == curr_stats.bin_edges[0] and updated_max == curr_stats.bin_edges[-1]:
                histogram_updates = curr_stats.histogram
                expanded_bin_edges = curr_stats.bin_edges
            else:
                expanded_bin_edges = self._create_bin_edges(updated_min, updated_max, input_tensor.device)
                histogram_updates = torch.zeros_like(curr_stats.histogram)
                src_bin_start = curr_stats.bin_edges[:-1]
                dest_bin_index = torch.bucketize(
                    src_bin_start, expanded_bin_edges[1:-1], right=True
                )
                # A wider destination grid overlaps each old bin in at most two places.
                width = curr_stats.bin_edges.diff()
                fraction = ((expanded_bin_edges[dest_bin_index + 1] - src_bin_start) / width).clamp(0, 1)
                left_count = (fraction * curr_stats.histogram).round().to(histogram_updates.dtype)
                histogram_updates.index_add_(0, dest_bin_index, left_count)
                next_bin = (dest_bin_index + 1).clamp(max=self.num_bins - 1)
                histogram_updates.index_add_(0, next_bin, curr_stats.histogram - left_count)

            if torch.equal(new_stats.bin_edges, expanded_bin_edges):
                batch_counts = new_stats.histogram
            else:
                batch_counts = self._count_bins(curr_input, expanded_bin_edges)
            expanded_histogram = histogram_updates + batch_counts
            self.stats[index] = Histogram(
                expanded_histogram, expanded_bin_edges,
                updated_min.to(new_stats.min.dtype), updated_max.to(new_stats.max.dtype)
            )

    def reset_stats(self):
        self.stats = [Histogram() for _ in self.stats]

    def get_stats(self) -> List[Histogram]:
        return self.stats


def adjust_min_max(curr_min, curr_max, num_steps, is_symmetric):
    minimum_scale = _get_minimum_scale(num_steps)

    # ensure that 0 is in the range
    curr_min = torch.minimum(curr_min, torch.zeros_like(curr_min))
    curr_max = torch.maximum(curr_max, torch.zeros_like(curr_max))

    # ensure that min/max are finite
    curr_min.clamp_(min=torch.finfo(curr_max.dtype).min, max=0)
    curr_max.clamp_(min=0, max=torch.finfo(curr_max.dtype).max)

    # ensure that min/max aren't too close
    tensor_threshold = (curr_max - curr_min) / num_steps

    if is_symmetric:
        curr_min[tensor_threshold < minimum_scale] -= minimum_scale * math.ceil(
            num_steps / 2
        )
        curr_max[tensor_threshold < minimum_scale] += minimum_scale * math.floor(
            num_steps / 2
        )
    else:
        curr_max[tensor_threshold < minimum_scale] += minimum_scale * num_steps

    if is_symmetric:
        num_pos_steps = math.floor(num_steps / 2)
        num_neg_steps = math.ceil(num_steps / 2)
        delta = max(curr_max / num_pos_steps, -curr_min / num_neg_steps)
        offset = -1 * num_neg_steps

        curr_min = offset * delta
        curr_max = num_pos_steps * delta

    return curr_min, curr_max


def validate_percentile(percentile):
    if not 50 <= float(percentile) <= 100:
        raise ValueError("Percentile value must be within 50-100 range")
    return float(percentile)


@torch.no_grad()
def percentile_ranges(stats, shape, num_steps, is_symmetric, percentile):
    """Compute ranges from accumulated counts using the same rule in both frontends."""
    if num_steps <= 0:
        raise ValueError(
            "The number of quantization bins cannot be less than or equal to 0."
        )

    if stats[0].histogram is None:
        raise RuntimeError("No statistics present to compute encodings.")

    encoding_min_list = []
    encoding_max_list = []

    for list_elem in stats:
        cum_sum = torch.cumsum(list_elem.histogram, dim=0, dtype=torch.float64)
        # trim percentile value from min and max
        max_index = torch.searchsorted(cum_sum, cum_sum[-1] * percentile / 100)
        min_index = torch.searchsorted(
            cum_sum, cum_sum[-1] * (1 - percentile / 100)
        )

        # The retained interval spans both percentile bins, including their far edges.
        left, right = (0, -1) if percentile == 100 else (min_index, max_index + 1)
        curr_min = list_elem.bin_edges[left].to(list_elem.min.dtype)
        curr_max = list_elem.bin_edges[right].to(list_elem.max.dtype)
        # adjust min/max
        updated_min, updated_max = adjust_min_max(
            curr_min, curr_max, num_steps, is_symmetric
        )
        encoding_min_list.append(updated_min)
        encoding_max_list.append(updated_max)

    encoding_min = torch.tensor(
        encoding_min_list,
        device=stats[0].histogram.device,
        dtype=stats[0].min.dtype,
    )
    encoding_min = torch.reshape(encoding_min, shape)

    encoding_max = torch.tensor(
        encoding_max_list,
        device=stats[0].histogram.device,
        dtype=stats[0].max.dtype,
    )
    encoding_max = torch.reshape(encoding_max, shape)

    return encoding_min, encoding_max


class PercentileCalibrator:
    """Streaming percentile statistics; raw observations are never retained."""

    def __init__(self, shape=(), *, num_bins=2048, percentile=100):
        if num_bins <= 0:
            raise ValueError("Number of bins cannot be less than or equal to 0.")
        self.observer = HistogramObserver(shape, num_bins)
        self.percentile = validate_percentile(percentile)

    def update_stats(self, tensor):
        if tensor.numel() == 0:
            return []
        stats = self.observer.collect_stats(tensor)
        self.observer.merge_stats(stats, tensor)
        return stats

    def reset_stats(self):
        self.observer.reset_stats()

    def compute_encodings(self, num_steps, is_symmetric):
        return percentile_ranges(self.observer.stats, self.observer.shape,
                                 num_steps, is_symmetric, self.percentile)
