from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from mv_recon.eval import claim_model_output_dir


class TestOutputIsolation(unittest.TestCase):
    def test_first_run_claims_plain_slug(self):
        with tempfile.TemporaryDirectory() as td:
            claimed = claim_model_output_dir(td, "model_ckpt10")
            self.assertEqual(claimed, os.path.join(td, "model_ckpt10"))
            self.assertTrue(os.path.isdir(claimed))

    def test_concurrent_collision_gets_pid_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            os.mkdir(os.path.join(td, "model_ckpt10"))
            with mock.patch("os.getpid", return_value=12345):
                claimed = claim_model_output_dir(td, "model_ckpt10")
            self.assertEqual(claimed, os.path.join(td, "model_ckpt10-pid12345"))
            self.assertTrue(os.path.isdir(claimed))


if __name__ == "__main__":
    unittest.main()
