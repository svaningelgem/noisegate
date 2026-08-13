"""Export DeepFilterNet3 to ONNX from the published checkpoint.

Runs upstream's own `df.scripts.export` — we are not reimplementing anything,
just driving it — with two environment workarounds:

1. `sqlite3` and `monkeytype` are imported *first*. `df` pulls in torch, which
   loads the system libstdc++; miniforge's libicui18n then wants a newer
   CXXABI than that provides, so a later `import sqlite3` dies with
   "CXXABI_1.3.15 not found". Importing it before torch gets in first and the
   whole chain works. The symptom is misleading: upstream guards its export
   with `import monkeytype`, so the failure surfaces as "Failed to import
   monkeytype", which is not the actual problem.

2. PYTHONPATH must point at the repo checkout rather than the installed df
   wheel, because the wheel does not ship a working scripts/export.py path for
   this checkpoint layout.

Usage:
    python export_dfn3.py <model_base_dir> <export_dir>

e.g.  python export_dfn3.py /root/dfn3ref/DeepFilterNet3 /root/dfn3export

Produces enc.onnx, erb_dec.onnx and df_dec.onnx: the encoder/decoder split.
The STFT front-end deliberately stays outside the graph — inputs are
`feat_erb` and `feat_spec`, which is what crates/dsp/src/dfn_frontend.rs
already computes, bit-exactly against libDF.
"""

import sqlite3  # noqa: F401  MUST come before df; see above
import monkeytype  # noqa: F401
import runpy
import sys

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(2)

model_base_dir, export_dir = sys.argv[1], sys.argv[2]
sys.argv = [
    "df.scripts.export",
    "--model-base-dir",
    model_base_dir,
    export_dir,
]
runpy.run_module("df.scripts.export", run_name="__main__")
