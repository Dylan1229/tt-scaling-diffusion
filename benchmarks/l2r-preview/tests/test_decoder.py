import pytest
import torch
from torch.nn import functional as F

from l2r_preview import L2RDecoder, WanLatentSpec, flow_prediction_to_clean
from l2r_preview.io import load_checkpoint, rgb_to_uint8, save_checkpoint
from l2r_preview.model import LearnedSpatialUpsample

torch.set_num_threads(2)


def test_shapes_and_parameter_budget():
    model = L2RDecoder()
    assert model.num_parameters == 59395
    assert model(torch.randn(1, 16, 2, 3, 4)).shape == (1, 3, 2, 24, 32)
    wan = L2RDecoder(48, 16)
    assert wan.num_parameters == 524995
    assert wan(torch.randn(1, 48, 1, 2, 3)).shape == (1, 3, 1, 32, 48)


def test_temporal_causality_and_chunk_equivalence():
    torch.manual_seed(1)
    model = L2RDecoder(3, 2).eval()
    x = torch.randn(1, 3, 12, 5, 7)
    changed = x.clone()
    changed[:, :, 6:] = torch.randn_like(changed[:, :, 6:]) * 100
    with torch.no_grad():
        result = model(x)
        assert torch.allclose(result[:, :, :6], model(changed)[:, :, :6], atol=1e-6, rtol=1e-5)
        chunked = torch.cat([out for _, out in model.iter_chunks(x, 2)], dim=2)
        assert torch.allclose(result, chunked, atol=1e-6, rtol=1e-5)


def test_upsample_initialization_does_not_mix_batch_channel_or_time():
    x = torch.arange(2 * 3 * 4 * 2 * 3, dtype=torch.float32).reshape(2, 3, 4, 2, 3)
    up = LearnedSpatialUpsample(3)
    expected = x.repeat_interleave(3, -2).repeat_interleave(3, -1)
    assert torch.equal(up(x), expected)


def test_wan_normalization_and_flow_sign():
    spec = WanLatentSpec(2, 8, 4, (0.3, -0.2), (0.4, 1.2))
    z = torch.randn(2, 2, 3, 4, 5)
    assert torch.allclose(spec.normalize(spec.denormalize(z)), z, atol=1e-6)
    assert spec.rgb_frame_indices(4).tolist() == [0, 4, 8, 12]
    noise = torch.randn_like(z)
    sigma = torch.tensor([0.2, 0.7])
    s = sigma.view(-1, 1, 1, 1, 1)
    sample = (1 - s) * z + s * noise
    assert torch.allclose(flow_prediction_to_clean(sample, noise - z, sigma), z, atol=1e-6)
    assert torch.equal(flow_prediction_to_clean(z, noise, 0.0), z)
    with pytest.raises(ValueError):
        flow_prediction_to_clean(z, noise, float("nan"))


def test_checkpoint_and_rgb_layout(tmp_path):
    torch.manual_seed(5)
    model = L2RDecoder(2, 2).eval()
    spec = WanLatentSpec(2, 2, 4, (0.1, 0.2), (1.0, 2.0))
    path = tmp_path / "decoder.pt"
    save_checkpoint(path, model, spec, {"steps": 3})
    restored, restored_spec, metadata = load_checkpoint(path)
    x = torch.randn(1, 2, 3, 4, 5)
    assert torch.equal(model(x), restored(x))
    assert restored_spec == spec and metadata["steps"] == 3
    rgb = torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1, 1)
    assert rgb_to_uint8(rgb).tolist() == [[[[0, 128, 255]]]]


def test_gradients_reach_upsampler_and_training_reduces_fixed_loss():
    torch.manual_seed(4)
    model = L2RDecoder(2, 2)
    x = torch.randn(1, 2, 2, 5, 5)
    target = torch.full((1, 3, 2, 10, 10), 0.25)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    initial = F.mse_loss(model(x), target).item()
    for _ in range(60):
        opt.zero_grad()
        loss = F.mse_loss(model(x), target)
        loss.backward()
        assert model.upsample.stencil.conv.weight.grad is not None
        opt.step()
    assert F.mse_loss(model(x), target).item() < initial * 0.2


@pytest.mark.parametrize("shape", [(1, 3, 2, 3), (1, 5, 2, 3, 4), (1, 2, 0, 3, 4)])
def test_bad_shapes_fail_explicitly(shape):
    with pytest.raises(ValueError):
        L2RDecoder(2, 2)(torch.empty(shape))
