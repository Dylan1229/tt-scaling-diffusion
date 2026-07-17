from ttsd.eval.vbench import parse_staged_video_stem


def test_parse_current_staging_name() -> None:
    assert parse_staged_video_stem("a person swimming-seed0012") == (
        "a person swimming",
        12,
    )


def test_parse_legacy_staging_name() -> None:
    assert parse_staged_video_stem("a person swimming-12") == (
        "a person swimming",
        12,
    )
