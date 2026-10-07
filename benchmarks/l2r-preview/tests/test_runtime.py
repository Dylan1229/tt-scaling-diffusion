import pytest
import torch
from l2r_preview.pyramid import PyramidPreviewDecoder
from l2r_preview.runtime import make_preview_decoder


def test_eager_runtime_preserves_rgb_contract_and_uses_new_input():
    torch.manual_seed(14)
    model=PyramidPreviewDecoder(width=8,blocks=1,stage_widths=(8,4,2))
    decode=make_preview_decoder(model)
    a=torch.randn(1,48,3,3,4)
    b=torch.randn_like(a)
    with torch.inference_mode():
        first=decode(a)
        retained=first.clone()
        second=decode(b)
        assert first.shape==(1,3,3,48,64)
        torch.testing.assert_close(first,model(a).add(1).mul(.5).clamp(0,1),rtol=0,atol=0)
        assert torch.equal(first,retained)
        assert not torch.equal(first,second)
        assert second.min()>=0 and second.max()<=1


def test_runtime_rejects_unsupported_modes_and_cpu_compile():
    model=PyramidPreviewDecoder(width=8,blocks=1,stage_widths=(8,4,2))
    with pytest.raises(ValueError,match="Unsupported"):
        make_preview_decoder(model,"invalid")
    with pytest.raises(ValueError,match="CUDA"):
        make_preview_decoder(model,"default")
