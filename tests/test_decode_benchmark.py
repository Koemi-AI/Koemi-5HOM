from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def load_decode_benchmark():
    module_path = REPOSITORY_ROOT / "benchmarks" / "run_decode_benchmark.py"
    specification = importlib.util.spec_from_file_location("run_decode_benchmark", module_path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class DecodeBenchmarkTests(unittest.TestCase):
    def test_the_benchmark_reports_every_cpu_path(self) -> None:
        module = load_decode_benchmark()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "decode.json"
            status = module.main(
                [
                    "--device",
                    "cpu",
                    "--embedding-size",
                    "16",
                    "--memory-features",
                    "4",
                    "--local-memory-size",
                    "4",
                    "--salience-memory-size",
                    "3",
                    "--scan-chunk",
                    "8",
                    "--max-new-tokens",
                    "4",
                    "--greedy",
                    "--synthetic-draft",
                    "--draft-length",
                    "2",
                    "--report",
                    str(report_path),
                ]
            )
            self.assertEqual(0, status)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        paths = {measurement["path"] for measurement in payload["measurements"]}
        self.assertEqual({"baseline", "fast_decode", "speculative_ngram", "speculative_model"}, paths)
        for measurement in payload["measurements"]:
            self.assertEqual(4, measurement["generated_tokens"])
            self.assertGreater(measurement["tokens_per_second"], 0.0)


if __name__ == "__main__":
    unittest.main()
