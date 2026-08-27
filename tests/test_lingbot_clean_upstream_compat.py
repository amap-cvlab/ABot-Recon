import torch
import torch.nn as nn

from models.fastmodel import LingBotMAP


def _wrapper(model):
    wrapper = LingBotMAP.__new__(LingBotMAP)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.mode = "streaming"
    wrapper.num_scale_frames = 8
    wrapper.keyframe_interval = 1
    wrapper.window_size = 64
    wrapper.overlap_size = 16
    return wrapper


class _CleanOfficialModel(nn.Module):
    def inference_streaming(
        self, images, *, num_scale_frames, keyframe_interval, output_device
    ):
        count = images.shape[0]
        return {
            "pose_enc": torch.zeros(1, count, 9),
            "depth": torch.arange(count).view(1, count, 1, 1, 1).float(),
        }


class _SelectiveDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.received_indices = None

    def inference_streaming(
        self,
        images,
        *,
        num_scale_frames,
        keyframe_interval,
        output_device,
        dense_output_indices=None,
    ):
        self.received_indices = dense_output_indices
        count = images.shape[0]
        dense_count = count if dense_output_indices is None else len(dense_output_indices)
        return {
            "pose_enc": torch.zeros(1, count, 9),
            "depth": torch.zeros(1, dense_count, 1, 1, 1),
        }


def test_clean_official_lingbot_runs_without_selective_dense_argument():
    output = _wrapper(_CleanOfficialModel())(
        torch.zeros(4, 3, 8, 8), dense_output_indices=[0, 2]
    )

    assert output["dense_output_indices_applied"] is False
    assert output["depth"].shape[1] == 4


def test_patched_lingbot_receives_selective_dense_indices():
    model = _SelectiveDenseModel()
    output = _wrapper(model)(
        torch.zeros(4, 3, 8, 8), dense_output_indices=[0, 2]
    )

    assert model.received_indices == [0, 2]
    assert output["dense_output_indices_applied"] is True
    assert output["depth"].shape[1] == 2
