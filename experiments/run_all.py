"""Run every experiment in order and record the total CPU time.

Usage:  python experiments/run_all.py [exp1 exp2 ...]
"""

from __future__ import annotations

import importlib
import json
import os
import resource
import sys
import time
from pathlib import Path

MODULES = ["exp1_calibration", "exp2_coverage", "exp3_modelclass",
           "exp4_resolution", "exp5_robustness", "exp6_dimension",
           "exp7_invariance", "exp8_tightness", "exp9_modelerror",
           "exp10_noise", "exp11_scaling", "exp12_realdata",
           "exp13_tolerance"]


def cpu_seconds():
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime)


def main():
    wanted = sys.argv[1:] or MODULES
    mods = [m for m in MODULES if any(w in m for w in wanted)]
    log = {}
    for name in mods:
        print(f"=== {name} ===", flush=True)
        c0, w0 = cpu_seconds(), time.time()
        importlib.import_module(name).main()
        log[name] = {"cpu_seconds": cpu_seconds() - c0, "wall_seconds": time.time() - w0}
        print(f"=== {name}: {log[name]['cpu_seconds']:.0f} CPU-s, "
              f"{log[name]['wall_seconds']:.0f} wall-s ===", flush=True)
    out = Path(__file__).resolve().parents[1] / "results" / "compute_log.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(log)
    prev["total_cpu_hours"] = sum(v["cpu_seconds"] for k, v in prev.items()
                                  if isinstance(v, dict)) / 3600.0
    out.write_text(json.dumps(prev, indent=1))
    print(json.dumps(prev, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
