import unittest
from unittest import mock

import torch

from models.fastmodel import HorizonStreamEval
from interfaces.horizonstream import _w2c_3x4_to_c2w_3x4


class HorizonStreamPrecisionTest(unittest.TestCase):
    def _adapter(self, amp_dtype):
        adapter = HorizonStreamEval.__new__(HorizonStreamEval)
        adapter.amp_dtype = amp_dtype
        return adapter

    def test_fp32_disables_cuda_autocast(self):
        enabled, dtype = self._adapter("fp32")._autocast_settings(
            torch.device("cuda:0")
        )
        self.assertFalse(enabled)
        self.assertEqual(dtype, torch.float32)

    def test_explicit_bf16_enables_cuda_autocast(self):
        enabled, dtype = self._adapter("bf16")._autocast_settings(
            torch.device("cuda:0")
        )
        self.assertTrue(enabled)
        self.assertEqual(dtype, torch.bfloat16)

    def test_cpu_always_disables_autocast(self):
        enabled, dtype = self._adapter("bf16")._autocast_settings(
            torch.device("cpu")
        )
        self.assertFalse(enabled)
        self.assertEqual(dtype, torch.float32)

    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 0))
    def test_auto_uses_bf16_on_ampere(self, _):
        enabled, dtype = self._adapter("auto")._autocast_settings(
            torch.device("cuda:0")
        )
        self.assertTrue(enabled)
        self.assertEqual(dtype, torch.bfloat16)

    @mock.patch("torch.cuda.get_device_capability", return_value=(7, 5))
    def test_auto_uses_fp16_before_ampere(self, _):
        enabled, dtype = self._adapter("auto")._autocast_settings(
            torch.device("cuda:0")
        )
        self.assertTrue(enabled)
        self.assertEqual(dtype, torch.float16)

    def test_bf16_w2c_is_promoted_before_inverse(self):
        w2c = torch.eye(4, dtype=torch.bfloat16)[None, :3]
        c2w = _w2c_3x4_to_c2w_3x4(w2c)
        self.assertEqual(c2w.dtype, torch.float32)
        torch.testing.assert_close(c2w, torch.eye(4)[None, :3])


if __name__ == "__main__":
    unittest.main()
