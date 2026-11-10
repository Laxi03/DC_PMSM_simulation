from contextlib import redirect_stderr
from io import StringIO
import unittest

from td3_drive_control.cli import main


class CLITests(unittest.TestCase):
    def test_invalid_options_fail_before_starting_work(self):
        cases = [
            (["train", "--motor", "dc", "--run", "../bad"], "Run name"),
            (["train", "--motor", "pmsm", "--run", "valid", "--steps", "0"], "steps"),
            (["baseline", "--motor", "dc", "--duration", "nan"], "Duration"),
            (["baseline", "--motor", "dc", "--perturbation", "R"], "only supported for PMSM"),
            (["baseline", "--motor", "pmsm", "--task", "speed", "--control-mode", "locked"], "only supported for current"),
        ]
        for argv, message in cases:
            with self.subTest(argv=argv), redirect_stderr(StringIO()) as errors:
                with self.assertRaises(SystemExit) as exit:
                    main(argv)
                self.assertEqual(exit.exception.code, 2)
                self.assertIn(message, errors.getvalue())
