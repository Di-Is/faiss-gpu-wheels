#!/usr/bin/env -S uv run --script -q # noqa: EXE003
"""Generate pyproject.toml.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

# /// script
# requires-python = "~=3.12"
# dependencies = [
#     "jinja2",
#     "packaging",
#     "click",
#     "pydanclick"
# ]
# ///
import subprocess
from pathlib import Path
from typing import Literal

import click
import tomllib
from jinja2 import Environment, FileSystemLoader
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydanclick import from_pydantic
from pydantic import BaseModel


def get_compatible_python(requires_python: str) -> list[str]:
    """Get compatible python versions.

    Args:
        requires_python: requires-python format str in pyproject.toml

    Returns:
        compatible python list
    """
    specifier = SpecifierSet(requires_python)
    compatible_versions = []
    minor = 0
    start = False
    while True:
        version = Version(f"3.{minor}")
        if version in specifier:
            compatible_versions.append(str(version))
            minor += 1
            start = True
        elif start:
            break
        else:
            minor += 1
    return compatible_versions


class Args(BaseModel):
    """Script argument."""

    variant: Literal["cpu", "gpu-cu11", "gpu-cu12"]


@click.command()
@from_pydantic(Args)
def cli(args: Args) -> None:
    """CLI function."""
    template_dir = Path(__file__).parent / "template"
    env = Environment(loader=FileSystemLoader(template_dir))  # noqa: S701
    template = env.get_template("pyproject.toml.j2")

    variant_path = Path(__file__).parent.parent / "variant" / f"faiss-{args.variant}"
    with (variant_path / "config.toml").open("rb") as f:
        config = tomllib.load(f)

    requires_python = config["python"]["requires-python"]
    classifiers = [
        f"Programming Language :: Python :: {v}" for v in get_compatible_python(requires_python)
    ]
    match args.variant:
        case "cpu":
            dependencies = []
            optional_deps = {}
        case "gpu-cu11":
            dependencies = ["nvidia-cuda-runtime-cu11", "nvidia-cublas-cu11"]
            optional_deps = {
                "fix-cuda": [
                    f"nvidia-cuda-runtime-cu11=={config["cuda"]["runtime-version"]}",
                    f"nvidia-cublas-cu11=={config["cuda"]["cublas-version"]}",
                ]
            }
            classifiers += [
                "Environment :: GPU :: NVIDIA CUDA",
                "Environment :: GPU :: NVIDIA CUDA :: 11",
            ]
        case "gpu-cu12":
            dependencies = ["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"]
            optional_deps = {
                "fix-cuda": [
                    f"nvidia-cuda-runtime-cu12=={config["cuda"]["runtime-version"]}",
                    f"nvidia-cublas-cu12=={config["cuda"]["cublas-version"]}",
                ]
            }
            classifiers += [
                "Environment :: GPU :: NVIDIA CUDA",
                "Environment :: GPU :: NVIDIA CUDA :: 12",
            ]
    context = {
        "requires_python": requires_python,
        "variant": args.variant,
        "dependencies": ",".join(f'"{s}"' for s in dependencies),
        "optional_dependencies": "\n".join(f"{k} = {v}" for k, v in optional_deps.items()),
        "classifiers": ",".join(f'"{s}"' for s in classifiers),
    }

    output = template.render(context)
    pyproject_path = variant_path / "pyproject.toml"
    pyproject_path.write_text(output)

    _ = subprocess.run(  # noqa: S603
        ["uv", "run", "taplo", "format", str(pyproject_path)],  # noqa: S607
        check=True,
        capture_output=True,
    )


if __name__ == "__main__":
    cli()
