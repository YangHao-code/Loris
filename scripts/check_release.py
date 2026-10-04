#!/usr/bin/env python3
"""Test and install a source snapshot without writing build files to the repo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new directory outside this checkout")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = Path(args.output).expanduser().resolve()
    if out == repo or repo in out.parents:
        raise ValueError("validation output must be outside the source checkout")
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    shutil.copytree(
        repo,
        source,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", "*.egg-info", "build", "dist", ".git"
        ),
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        LORIS_SSL_SOURCE=str(source / "backends/transformer/src"),
    )
    env.pop("PYTHONPATH", None)
    report = {"status": "running", "python": sys.version, "source_files": {}}
    for path in sorted(source.rglob("*")):
        if path.is_file():
            report["source_files"][str(path.relative_to(source))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    def save():
        (out / "VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, argv, cwd, environment=env):
        print(name, flush=True)
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(
                argv,
                cwd=str(cwd),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    save()
    try:
        run(
            "tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-rs",
                "--junitxml=" + str(out / "tests.xml"),
            ],
            source,
        )
        tree = ET.parse(str(out / "tests.xml"))
        cases = list(tree.iter("testcase"))
        report["tests"] = dict(
            total=len(cases),
            failures=len(list(tree.iter("failure"))),
            errors=len(list(tree.iter("error"))),
            skipped=[
                {
                    "test": node.get("classname") + "." + node.get("name"),
                    "reason": node.find("skipped").get("message"),
                }
                for node in cases
                if node.find("skipped") is not None
            ],
        )
        run(
            "build",
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(out / "dist"),
                str(source),
            ],
            out,
        )
        (wheel,) = (out / "dist").glob("loris-*.whl")
        with zipfile.ZipFile(str(wheel)) as archive:
            names = set(archive.namelist())
        expected = {str(p.relative_to(source)) for p in (source / "loris").rglob("*.py")}
        expected.update(
            str(p.relative_to(source / "backends/transformer/src"))
            for p in (source / "backends/transformer/src").rglob("*.py")
        )
        if expected - names:
            raise RuntimeError("wheel omits source modules: " + ", ".join(sorted(expected - names)))
        report["wheel"] = {
            "file": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "python_modules_verified": len(expected),
        }
        installed = out / "installed"
        run(
            "install",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(installed),
                str(wheel),
            ],
            out,
        )
        check = out / "installed_check.py"
        check.write_text(
            """import importlib, json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1]).resolve()
modules = ['loris', 'loris._engine.typed_runtime', 'loris._storage.discovery_factory',
           'loris.discovery.runner', 'loris.discovery.search.execution',
           'loris.labeling.runner', 'loris.experiments.workflow', 'loris_ssl.chase_backend']
for name in modules:
    module = importlib.import_module(name)
    assert root in pathlib.Path(module.__file__).resolve().parents, (name, module.__file__)
all_modules = []
for path in sorted((root / 'loris').rglob('*.py')):
    parts = list(path.relative_to(root).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    name = '.'.join(parts)
    module = importlib.import_module(name)
    assert root in pathlib.Path(module.__file__).resolve().parents, (name, module.__file__)
    all_modules.append(name)
for name in ['loris', 'loris.discovery', 'loris.labeling', 'loris.experiments']:
    subprocess.run([sys.executable, '-m', name, '--help'], check=True)
for name in ['loris-discover', 'loris-label', 'loris-experiments']:
    subprocess.run([str(root / 'bin' / name), '--help'], check=True)
print(json.dumps({'installed_imports': all_modules, 'cli_help': 'passed', 'console_scripts': 'passed'}))
"""
        )
        installed_env = dict(env, PYTHONPATH=str(installed), LORIS_SSL_SOURCE=str(installed))
        run("installed_cli", [sys.executable, str(check), str(installed)], out, installed_env)
        # Source-independent semantic checks catch modules missing only on less
        # common inference paths. Tests are copied without the source package.
        semantic = out / "installed_semantics"
        semantic.mkdir()
        shutil.copytree(
            source / "tests", semantic / "tests", ignore=shutil.ignore_patterns("__pycache__")
        )
        # These public preparation/comparison tools intentionally ship in the
        # source release. Their tests still import the installed runtime only.
        shutil.copytree(
            source / "scripts", semantic / "scripts", ignore=shutil.ignore_patterns("__pycache__")
        )
        selected = [
            "tests/test_paper_examples.py",
            "tests/test_incremental_release.py",
            "tests/test_release_inference_properties.py",
            "tests/test_direct_rule_precision.py",
            "tests/test_direct_rule_discount.py",
            "tests/test_receiver_mapping.py",
            "tests/test_model_arbitration.py",
            "tests/test_conflict_quarantine.py",
            "tests/test_conflict_protocol_loop.py",
            "tests/test_quarantine_reliability.py",
            "tests/test_quarantine_llm_guard.py",
            "tests/test_relation_rule_reliability.py",
            "tests/test_feedback_check_contract.py",
            "tests/test_author_rule_forms.py",
            "tests/test_relation_propagation_rdg.py",
            "tests/test_relation_storage.py",
            "tests/test_bo_rule_forms.py",
            "tests/test_paper_baseline_round_snapshots.py",
            "tests/test_probability_lineage_fields.py",
            "tests/test_compat_public_api.py",
            "tests/test_selector_checkpoint_compat.py",
            "tests/test_public_probability_predicates.py",
            "tests/test_optional_model_imports.py",
            "tests/test_comparison_knowledge_semantics.py",
            "tests/test_public_model_semantics.py",
            "tests/test_perturbation_interfaces.py",
            "tests/test_structured_checkpoint_paths.py",
            "tests/test_head_tail_collator.py",
            "tests/test_update_timeout_boundaries.py",
            "tests/test_unknown_incremental.py",
            "tests/test_retained_labeling_integration.py",
            "tests/test_retained_calibration.py",
            "tests/test_full_dataset_preparation.py",
            "tests/test_discovery_expanded_space.py",
            "tests/test_discovery_stage_plan.py",
            "tests/test_discovery_metadata_projection.py",
            "tests/test_workflow_calibration_semantics.py",
            "tests/test_zero_fit_initializer.py",
        ]
        run(
            "installed_semantics",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--junitxml=" + str(out / "installed_semantics.xml"),
            ]
            + selected,
            semantic,
            installed_env,
        )
        installed_tree = ET.parse(str(out / "installed_semantics.xml"))
        report["installed_semantics"] = dict(
            status="passed",
            total=len(list(installed_tree.iter("testcase"))),
            failures=len(list(installed_tree.iter("failure"))),
            errors=len(list(installed_tree.iter("error"))),
            skipped=len(list(installed_tree.iter("skipped"))),
        )
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__, reason=str(error))
        raise
    finally:
        save()
    print(str(out / "VALIDATION.json"), flush=True)


if __name__ == "__main__":
    main()
