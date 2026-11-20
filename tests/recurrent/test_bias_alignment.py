"""Independent recurrent bias grids must rescale into their GEMM accumulator."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from quant_gru import QuantGRU
from quant_lstm import QuantLSTM


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RecurrentBiasAlignmentTest(unittest.TestCase):
    def export(self, model, path):
        if isinstance(model, QuantLSTM):
            return model.export_quant_params()
        model.export_quant_params(str(path))
        return json.loads(path.read_text())

    def load(self, model, document, path):
        if isinstance(model, QuantLSTM):
            model.load_quant_params(document)
        else:
            path.write_text(json.dumps(document))
            model.load_quant_params(str(path))
        model.use_quantization = True

    def set_grid(self, record, scales):
        count = len(record["scale"]) if isinstance(record["scale"], list) else None
        values = [scales[index % len(scales)] for index in range(count or 1)]
        bits = int(record["dtype"].removeprefix("UINT").removeprefix("INT"))
        lower, upper = ((0, 2 ** bits - 1) if record["dtype"].startswith("U")
                        else (-2 ** (bits - 1), 2 ** (bits - 1) - 1))
        for key, items in (("scale", values), ("zero_point", [0] * len(values)),
                           ("real_min", [value * lower for value in values]),
                           ("real_max", [value * upper for value in values])):
            record[key] = items if count else items[0]

    def fixed_document(self, model, path, pot2):
        document = self.export(model, path)
        document["model_info"]["use_pot2_scale"] = pot2
        if "execution_metadata" in document:
            document["execution_metadata"]["standard_scale_mode"] = "pot2" if pot2 else "affine"
        for direction in ("operators", "operators_reverse"):
            if direction not in document:
                continue
            for name, record in document[direction].items():
                scales = [1 / 32]
                if name == "input":
                    scales = [1 / 8]
                elif name == "output":
                    scales = [1 / 16]
                elif name == "weight_ih":
                    scales = [1 / 4, 1 / 2]
                elif name == "weight_hh":
                    scales = [1 / 2, 1]
                elif name in ("bias_ih", "bias_hh"):
                    scales = [1 / 32, 1 / 16]
                elif name.endswith("gate_output") or name in ("cell_tanh", "cell_tanh_output"):
                    scales = [1 / 128]
                self.set_grid(record, scales)
            operators = document[direction]
            for bias, weight, activation in (("bias_ih", "weight_ih", "input"),
                                              ("bias_hh", "weight_hh", "output")):
                accumulator_scales = (torch.tensor(operators[weight]["scale"])
                                      * operators[activation]["scale"])
                torch.testing.assert_close(torch.tensor(operators[bias]["scale"]),
                                           accumulator_scales, rtol=0, atol=0)
        return document

    def assert_result_equal(self, actual, expected):
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        actual_states = actual[1] if isinstance(actual[1], tuple) else (actual[1],)
        expected_states = expected[1] if isinstance(expected[1], tuple) else (expected[1],)
        for actual_state, expected_state in zip(actual_states, expected_states):
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)

    def test_independent_bias_grids_match_accumulator_grids_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encodings.json"
            for cls in (QuantGRU, QuantLSTM):
                devices = ["cuda"]
                for device in devices:
                    for bidirectional in (False, True):
                        for pot2 in (False, True):
                            model = cls(2, 2, bidirectional=bidirectional).to(device).eval()
                            model.set_all_bitwidth(8)
                            with torch.no_grad():
                                for name, parameter in model.named_parameters():
                                    value = .375 if "bias" in name else .25
                                    if "bias" in name and "reverse" in name:
                                        value = .75
                                    parameter.fill_(-value if "reverse" in name else value)
                                inputs = torch.tensor([[[.25, -.125]], [[.5, .25]],
                                                       [[-.125, .375]]], device=device)
                                with model.calibration_context():
                                    model(inputs)
                            baseline = self.fixed_document(model, path, pot2)
                            self.load(model, baseline, path)
                            carriers = ("float32", "int32") if cls is QuantGRU else (None,)
                            for carrier in carriers:
                                if carrier is not None:
                                    model.quant_storage_dtype = carrier
                                with torch.no_grad():
                                    expected = model(inputs)
                                self.assertGreater(expected[0].abs().max().item(), 0)
                                for branch in ("bias_ih", "bias_hh"):
                                    with self.subTest(operator=cls.__name__, device=device,
                                                      bidirectional=bidirectional, pot2=pot2,
                                                      carrier=carrier, branch=branch):
                                        for suffix in ("", "_reverse")[:2 if bidirectional else 1]:
                                            parameter = getattr(model, branch + "_l0" + suffix)
                                            saved = parameter.detach().clone()
                                            with torch.no_grad():
                                                parameter.zero_()
                                                without_bias = model(inputs)[0]
                                                parameter.copy_(saved)
                                            self.assertFalse(torch.equal(without_bias, expected[0]))
                                        independent = copy.deepcopy(baseline)
                                        for index, direction in enumerate(("operators", "operators_reverse")):
                                            if direction in independent:
                                                factor = (2 if pot2 else 1.5) * (index + 1)
                                                self.set_grid(independent[direction][branch],
                                                              [factor / 32, factor / 16])
                                                value = .375 if index == 0 else -.75
                                                codes = value / torch.tensor(
                                                    independent[direction][branch]["scale"])
                                                torch.testing.assert_close(codes, codes.round(),
                                                                           rtol=0, atol=0)
                                                self.assertTrue(torch.all(codes.abs() <= 127))
                                        self.load(model, independent, path)
                                        with torch.no_grad():
                                            self.assert_result_equal(model(inputs), expected)
                                        exported = self.export(model, path)
                                        for direction in ("operators", "operators_reverse"):
                                            if direction in independent:
                                                self.assertEqual(exported[direction][branch],
                                                                 independent[direction][branch])
                                        restored = cls(2, 2, bidirectional=bidirectional).to(device).eval()
                                        restored.load_state_dict(model.state_dict())
                                        self.load(restored, exported, path)
                                        if carrier is not None:
                                            restored.quant_storage_dtype = carrier
                                        with torch.no_grad():
                                            self.assert_result_equal(restored(inputs), expected)
                                        self.load(model, baseline, path)


if __name__ == "__main__":
    unittest.main()
