"""Machine-enforced invariant: estimators never call estimators.

The whole point of phases 1-2 was to remove estimator->estimator layering: a
shared fit is a pure ``tools/`` function, and an experiment's Dataset-shaped
stage helpers (``track_flux_peaks``, ``state_iq_arrays``, ...) may be imported
as plain FUNCTIONS by a sibling composite — but no estimator module may import
another estimator subpackage's ``*Estimator`` class and drive it.

This walks the estimator subpackage sources statically and fails if any module
imports an ``*Estimator`` name from a *different* estimator subpackage. It is
the guard that keeps the layering from silently creeping back.
"""

import ast
from pathlib import Path

import scqat.estimators as estimators_pkg

ESTIMATORS_DIR = Path(estimators_pkg.__file__).parent


def _cross_estimator_class_imports(py: Path, own_subpkg: str):
    """Names ending in 'Estimator' imported from a DIFFERENT estimator
    subpackage by the module at ``py``."""
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    offences = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        parts = node.module.split(".")
        # scqat.estimators.<subpkg>[...]
        if parts[:2] != ["scqat", "estimators"] or len(parts) < 3:
            continue
        target_subpkg = parts[2]
        if target_subpkg == own_subpkg:
            continue  # same subpackage: fine
        for alias in node.names:
            if alias.name.endswith("Estimator"):
                offences.append(f"{node.module}.{alias.name}")
    return offences


def test_no_estimator_imports_another_estimator_class():
    subpackages = [
        d for d in ESTIMATORS_DIR.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
    ]
    assert subpackages, "no estimator subpackages found — path wrong?"

    violations = {}
    for subpkg in subpackages:
        for py in subpkg.glob("*.py"):
            offences = _cross_estimator_class_imports(py, subpkg.name)
            if offences:
                violations[str(py.relative_to(ESTIMATORS_DIR))] = offences

    assert not violations, (
        "estimators must never import another estimator's class (import the "
        "shared reduction from tools/ or a stage FUNCTION instead):\n"
        + "\n".join(f"  {f}: {imps}" for f, imps in sorted(violations.items()))
    )
