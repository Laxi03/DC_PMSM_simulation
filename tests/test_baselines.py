import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from td3_drive_control.cli import main
from td3_drive_control.evaluation import baseline, metrics
from td3_drive_control.experiments import Experiment


class BaselineTests(unittest.TestCase):
    def test_all_baseline_tasks_produce_finite_physical_metrics(self):
        for motor in ("dc", "pmsm"):
            for task in ("current", "speed"):
                with self.subTest(motor=motor, task=task):
                    case = Experiment(motor, task)
                    trace = baseline(case, duration=0.02, reference=None, random=False)
                    summary = metrics(trace, task)
                    self.assertEqual(summary["tracking_unit"], "rad/s" if task == "speed" else "A")
                    self.assertGreater(summary["samples"], 1)
                    self.assertAlmostEqual(summary["simulated_seconds"], 0.02)

    def test_random_baselines_repeat_by_seed(self):
        for motor in ("dc", "pmsm"):
            case = Experiment(motor, "speed", seed=76)
            first = baseline(case, duration=0.02, reference=None, random=True)
            second = baseline(case, duration=0.02, reference=None, random=True)
            for name in first.columns:
                np.testing.assert_array_equal(first.columns[name], second.columns[name])

    def test_pmsm_perturbation_is_visible_in_trace(self):
        case = Experiment("pmsm", "current", perturbation="R", relative_change=0.3, perturbation_time=0.01)
        trace = baseline(case, duration=0.02, reference=None, random=False)
        resistance = trace.columns["R"]
        self.assertAlmostEqual(resistance[-1] / resistance[0], 1.3)

    def test_cli_saves_portable_report(self):
        with TemporaryDirectory() as workspace:
            main(["baseline", "--motor", "dc", "--task", "speed", "--duration", "0.02",
                  "--workspace", workspace])
            folder, = (Path(workspace) / "outputs" / "dc" / "speed").iterdir()
            self.assertEqual({p.name for p in folder.iterdir()},
                             {"tracking.png", "timeseries.csv", "metrics.json", "manifest.json"})
            summary = json.loads((folder / "metrics.json").read_text())
            self.assertEqual(summary["ended_by"], "duration")
