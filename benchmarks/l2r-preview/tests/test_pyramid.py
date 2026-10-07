import torch
from torch.nn import functional as F

from l2r_preview.pyramid import PyramidPreviewDecoder, SubpixelUp


def small_model():
    torch.manual_seed(11)
    model = PyramidPreviewDecoder(latent_channels=3, width=8, blocks=2, stage_widths=(6, 4, 2))
    # Exercise nonzero past-frame connections, not just the initial identity taps.
    with torch.no_grad():
        for block in model.trunk:
            block.temporal.weight[:, :, :2].normal_(std=0.2)
    return model


def test_shapes_and_model_metadata():
    for variant in ("compact", "balanced"):
        model = PyramidPreviewDecoder.from_variant(variant)
        assert model(torch.randn(1, 48, 1, 2, 3)).shape == (1, 3, 1, 32, 48)
        assert model.num_parameters < 2_000_000
        assert model.temporal_context == 2*model.blocks
        assert model.spatial_halo == 2*model.blocks+4


def test_causal_chunking_and_batch_isolation():
    model = small_model().eval()
    x = torch.randn(2, 3, 9, 3, 4)
    with torch.inference_mode():
        whole = model(x)
        changed = x.clone()
        changed[0, :, 5:] = 100*torch.randn_like(changed[0, :, 5:])
        changed[1] = 100*torch.randn_like(changed[1])
        altered = model(changed)
        torch.testing.assert_close(whole[0, :, :5], altered[0, :, :5], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(whole[:1], model(x[:1]), rtol=1e-5, atol=1e-6)
        chunked = torch.cat([v for _, v in model.iter_chunks(x, 2)], dim=2)
        torch.testing.assert_close(whole, chunked, rtol=1e-5, atol=1e-6)


def test_initial_subpixel_phases_match_nearest_neighbor():
    torch.manual_seed(5)
    up = SubpixelUp(3, 4)
    x = torch.randn(2, 3, 5, 7)
    base = F.conv2d(x, up.projection.weight[::4], up.projection.bias[::4], padding=1)
    expected = base.repeat_interleave(2, -2).repeat_interleave(2, -1)
    torch.testing.assert_close(up(x), expected)


def test_spatial_halo_preserves_interior_prediction():
    model = small_model().eval()
    x = torch.randn(1, 3, 2, 26, 30)
    halo = model.spatial_halo
    y, xx, h, w = 11, 12, 2, 3
    with torch.inference_mode():
        full = model(x)
        patch = model(x[:, :, :, y-halo:y+h+halo, xx-halo:xx+w+halo])
        expected = full[..., y*16:(y+h)*16, xx*16:(xx+w)*16]
        actual = patch[..., halo*16:(halo+h)*16, halo*16:(halo+w)*16]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_backward_reaches_temporal_and_upsampling_weights():
    model = small_model()
    x = torch.randn(1, 3, 3, 4, 5)
    prediction = model(x)
    F.mse_loss(prediction, torch.randn_like(prediction)).backward()
    selected = [model.stem.weight, model.trunk[0].temporal.weight,
                model.stages[0][0].projection.weight, model.rgb.projection.weight]
    for weight in selected:
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0

def test_selected_frames_match_full_forward_and_gradients():
    model = small_model()
    x = torch.randn(1, 3, 7, 3, 4)
    full = model(x)[:, :, 4:6]
    selected = model(x, output_start=4, output_frames=2)
    torch.testing.assert_close(selected, full)
    weights = tuple(model.parameters())
    expected = torch.autograd.grad(full.square().mean(), weights)
    actual = torch.autograd.grad(selected.square().mean(), weights)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_pyramid_patch_plan_and_checkpoint_round_trip(tmp_path):
    import random

    from l2r_preview.io import load_checkpoint, save_checkpoint
    from l2r_preview.latents import WanLatentSpec
    from l2r_preview.openvid import crop_prediction, patch_plan
    model = small_model().eval()
    spec = WanLatentSpec(3, 16, 4, (0., 0., 0.), (1., 1., 1.))
    path = tmp_path/"pyramid.pt"
    save_checkpoint(path, model, spec, {"step": 1})
    restored, loaded_spec, meta = load_checkpoint(path)
    assert loaded_spec == spec and meta["step"] == 1
    x = torch.randn(1, 3, 8, 20, 22)
    with torch.inference_mode():
        full = model(x)
        torch.testing.assert_close(restored(x), full)
        for seed in (3, 9):
            plan = patch_plan(x.shape[1:], 16, random.Random(seed), core=(2, 3),
                              context=model.temporal_context, halo=model.spatial_halo)
            t0, t1, y0, y1, x0, x1 = plan["latent_slice"]
            offset, count, y, xx, h, w = plan["roi"]
            actual = crop_prediction(model(x[:, :, t0:t1, y0:y1, x0:x1],
                                            output_start=offset, output_frames=count),
                                     (0, count, y, xx, h, w))
            y, xx, h, w = plan["rgb_crop"]
            t = plan["latent_start"]
            torch.testing.assert_close(actual, full[:, :, t:t+count, y:y+h, xx:xx+w],
                                       rtol=2e-5, atol=2e-6)
