from pathlib import Path

from sfm_reconstruction.blender_viewer import build_blender_command


def test_build_blender_command_contains_importer_arguments() -> None:
    command = build_blender_command(
        Path("blender"),
        Path("results"),
        Path("results/reconstruction.blend"),
        trim_percentile=1.0,
        point_size=0.02,
    )

    assert command[:3] == ["blender", "--background", "--factory-startup"]
    assert "--python" in command
    assert command[-4:] == [
        "--trim-percentile",
        "1.0",
        "--point-size",
        "0.02",
    ]
    assert "results" in command
