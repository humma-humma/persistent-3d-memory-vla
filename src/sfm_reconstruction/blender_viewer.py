from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_WINDOWS_BLENDER = Path(
    r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
)


def find_blender(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"Blender executable not found: {path}")

    executable = shutil.which("blender")
    if executable:
        return Path(executable).resolve()
    if DEFAULT_WINDOWS_BLENDER.is_file():
        return DEFAULT_WINDOWS_BLENDER
    raise FileNotFoundError(
        "Blender was not found. Pass its executable with --blender."
    )


def importer_script() -> Path:
    script = Path(__file__).resolve().parents[2] / "scripts" / (
        "blender_import_reconstruction.py"
    )
    if not script.is_file():
        raise FileNotFoundError(f"Blender importer script not found: {script}")
    return script


def build_blender_command(
    blender: Path,
    result_dir: Path,
    output: Path,
    trim_percentile: float,
    point_size: float | None,
) -> list[str]:
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(importer_script()),
        "--",
        "--result-dir",
        str(result_dir),
        "--output",
        str(output),
        "--trim-percentile",
        str(trim_percentile),
    ]
    if point_size is not None:
        command.extend(["--point-size", str(point_size)])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Blender scene for an SfM reconstruction."
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--trim-percentile", type=float, default=1.0)
    parser.add_argument("--point-size", type=float)
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_after",
        help="Open the generated scene in Blender.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result_dir = args.result_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else result_dir / "reconstruction.blend"
    )
    blender = find_blender(args.blender)
    command = build_blender_command(
        blender,
        result_dir,
        output,
        args.trim_percentile,
        args.point_size,
    )
    subprocess.run(command, check=True)
    if not output.is_file():
        temporary_output = output.with_name(output.name + "@")
        detail = (
            f" Blender left a temporary file at {temporary_output}."
            if temporary_output.is_file()
            else ""
        )
        raise RuntimeError(f"Blender did not create {output}.{detail}")
    print(f"Saved Blender scene: {output}")
    if args.open_after:
        subprocess.Popen([str(blender), str(output)])


if __name__ == "__main__":
    main()
