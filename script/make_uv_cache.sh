#!/bin/bash
export UV_CACHE_DIR='/tmp/.cache/uv'
uv_bin="$(which uv)"
python_vers=(3.9 3.10 3.11 3.12)
variants=("cpu" "gpu-cu11" "gpu-cu12")
for var in ${python_vers[@]}; do
    for variant in ${variants[@]}; do
        sudo -E $uv_bin sync --python $var --no-install-project --project variant/$variant
        sudo rm -r variant/$variant/.venv
    done
done
