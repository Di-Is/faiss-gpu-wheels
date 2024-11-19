"""Serve test function using dagger pipeline.

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from dagger import Directory, dag

from ._type import Config
from ._util import install_uv


async def execute_test(source: Directory, wheel_dir: Directory, config: Config) -> None:
    """Execute test.

    Args:
        source: repository root
        wheel_dir: directory included wheels
        config: config file
    """
    for test_config in config.python.test:
        ctr = dag.container().from_(test_config.image)
        if "gpu-cu" in config.variant:
            ctr = ctr.experimental_with_gpu(["0"])
        for py_version in test_config.target_python_versions:
            wheel_name = (await wheel_dir.glob(f"*cp{py_version.replace(".", "")}*.whl"))[0]
            await (
                install_uv(ctr)
                .with_directory("/project", source, include=["faiss"])
                .with_workdir("/project")
                .with_env_variable("UV_SYSTEM_PYTHON", "1")
                .with_env_variable("OMP_NUM_THREADS", "1")
                .with_file(wheel_name, wheel_dir.file(wheel_name))
                .with_exec(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        py_version,
                        wheel_name,
                        *test_config.install_command,
                    ]
                )
                .with_exec(["uv", "run", "--python", py_version, *test_config.test_command])
                .sync()
            )
