from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from td3_drive_control.evaluation import _validate_checkpoint
from td3_drive_control.experiments import Experiment, environment_factory


@unittest.skipUnless(find_spec("gymnasium"), "Optional RL dependencies are not installed")
class EnvironmentTests(unittest.TestCase):
    def test_original_shapes_and_seeded_steps(self):
        shapes = {("dc", "current"): (6, 1), ("dc", "speed"): (6, 1),
                  ("pmsm", "current"): (9, 2), ("pmsm", "speed"): (11, 2)}
        for (motor, task), (observations, actions) in shapes.items():
            with self.subTest(motor=motor, task=task):
                env = environment_factory(Experiment(motor, task), training=False)()
                try:
                    self.assertEqual(env.observation_space.shape, (observations,))
                    self.assertEqual(env.action_space.shape, (actions,))
                    action = np.full(actions, 0.1, dtype=np.float32)
                    first, _ = env.reset(seed=123)
                    a, reward_a, _, _, _ = env.step(action)
                    second, _ = env.reset(seed=123)
                    b, reward_b, _, _, _ = env.step(action)
                    np.testing.assert_array_equal(first, second)
                    np.testing.assert_array_equal(a, b)
                    self.assertTrue(env.observation_space.contains(a))
                    self.assertEqual(reward_a, reward_b)
                finally:
                    env.close()

    def test_pmsm_training_randomization_remains_training_only(self):
        case = Experiment("pmsm", "current")
        train = environment_factory(case, training=True)()
        evaluate = environment_factory(case, training=False)()
        try:
            self.assertTrue(train.training_mode)
            self.assertTrue(train.train_randomization.enabled)
            self.assertFalse(evaluate.training_mode)
        finally:
            train.close()
            evaluate.close()


class CheckpointManifestTests(unittest.TestCase):
    def test_equal_shape_wrong_task_still_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            checkpoint = folder / "model.zip"
            checkpoint.touch()
            (folder / "manifest.json").write_text(json.dumps({
                "schema": "td3-drive-control/training-v1",
                "status": "complete",
                "experiment": {"motor": "dc", "task": "current"},
            }))
            with self.assertRaisesRegex(ValueError, "motor/task"):
                _validate_checkpoint(Experiment("dc", "speed"), checkpoint)

    def test_missing_checkpoint_is_actionable(self):
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "Checkpoint not found"):
                _validate_checkpoint(Experiment("dc", "speed"), Path(temp) / "missing.zip")
