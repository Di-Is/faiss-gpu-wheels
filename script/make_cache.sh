python_vers=(3.9 3.10 3.11 3.12)
variants=("cpu" "gpu-cu11" "gpu-cu12")
for var in ${python_vers[@]}; do
    for variant in ${variants[@]}; do
        echo "Building $variant with Python $var"
        uv sync --python $var --no-install-project --project variant/$variant
        rm -r variant/$variant/.venv
    done
done
