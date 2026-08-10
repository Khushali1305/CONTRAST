#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "$script_dir/.." && pwd)"
data_dir="${1:-$(cd "$package_root/.." && pwd)/CONTRAST_data}"
output_dir="${2:-$package_root/recomputed}"
python_bin="${PYTHON:-python3}"

cd "$package_root"
"$python_bin" -m unittest discover -s evaluation/tests -v
"$python_bin" -m evaluation.validate --data-dir "$data_dir"
"$python_bin" -m evaluation.aggregate --data-dir "$data_dir" --output-dir "$output_dir"
"$python_bin" -m evaluation.validate --data-dir "$data_dir" --recomputed-dir "$output_dir"

