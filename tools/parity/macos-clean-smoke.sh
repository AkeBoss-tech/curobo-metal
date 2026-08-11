#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
smoke_dir=$(mktemp -d "${TMPDIR:-/tmp}/curobo-metal-smoke.XXXXXX")
trap 'rm -rf "$smoke_dir"' EXIT HUP INT TERM

smoke_python=${CUROBO_METAL_SMOKE_PYTHON:-python3}
case "$smoke_python" in
    */*) smoke_python=$(CDPATH= cd -- "$(dirname -- "$smoke_python")" && pwd)/$(basename -- "$smoke_python") ;;
esac
venv_flags=
install_flags=
if [ "${CUROBO_METAL_SMOKE_USE_SYSTEM_PACKAGES:-0}" = "1" ]; then
    uv build --wheel --sdist --out-dir "$smoke_dir/dist" "$repo_dir"
    mkdir "$smoke_dir/site"
    "$smoke_python" -m zipfile -e "$smoke_dir"/dist/*.whl "$smoke_dir/site"
    cd "$smoke_dir"
    "$smoke_python" -I -c "import sys; sys.path.insert(0, '$smoke_dir/site'); import curobo_metal; from curobo_metal.ops.kinematics import forward_kinematics; print(curobo_metal.__file__)"
    exit
fi
"$smoke_python" -m venv $venv_flags "$smoke_dir/venv"
"$smoke_dir/venv/bin/python" -m pip install --upgrade build pip
"$smoke_dir/venv/bin/python" -m build --wheel --sdist --outdir "$smoke_dir/dist" "$repo_dir"
"$smoke_dir/venv/bin/python" -m pip install $install_flags "$smoke_dir"/dist/*.whl
cd "$smoke_dir"
"$smoke_dir/venv/bin/python" "$repo_dir/tools/packaging/wheel_smoke.py"
"$smoke_dir/venv/bin/python" -m pip check
