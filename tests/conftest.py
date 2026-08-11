"""Test helpers: load the pipeline scripts by path.

The scripts are standalone command-line programs, not an installed package, so
they are imported here by file path rather than by module name.
"""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODULES = {
    "stage1": REPO_ROOT / "code" / "stage1" / "stage1_placement.py",
    "stage2": REPO_ROOT / "code" / "stage2" / "stage2_run.py",
    "analysis": REPO_ROOT / "code" / "analysis" / "stage2_analyze_full.py",
}


def load(name):
    """Import one pipeline module by path.

    Raises ImportError with a readable message if the GPU stack (unsloth, trl) is
    absent -- tests that only need pure-Python logic skip themselves instead of
    failing, so the CPU self-tests remain runnable on a laptop.
    """
    path = MODULES[name]
    spec = importlib.util.spec_from_file_location(f"_pipeline_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
