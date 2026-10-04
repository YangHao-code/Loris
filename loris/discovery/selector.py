"""Launch the unchanged aligned selector with deterministic CUDA configured.

This module uses only stdlib before configuring the environment. Importing the
parent loris package can import torch for its compatibility shim; that does not
initialize CUDA. A live CUDA context is checked before any workspace change.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid


WORKSPACES = (":4096:8", ":16:8")


def configure_workspace(device, requested=None):
    """Set cuBLAS before CUDA initialization; never weaken determinism."""
    if requested is not None and requested not in WORKSPACES:
        raise ValueError("cublas_workspace must be :4096:8 or :16:8")
    if device != "cpu" and not (
        isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ValueError("selector requires explicit cpu or cuda:index")
    before = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device == "cpu":
        return dict(
            device=device,
            before=before,
            effective=before,
            requested=requested,
            action="cpu_environment_unchanged",
            cuda_initialized_before=None,
        )
    if before is not None and before not in WORKSPACES:
        raise ValueError(
            "invalid existing CUBLAS_WORKSPACE_CONFIG; explicitly correct the launch environment"
        )
    torch = sys.modules.get("torch")
    initialized = bool(torch is not None and torch.cuda.is_initialized())
    effective = requested or before or WORKSPACES[0]
    if initialized and before != effective:
        raise RuntimeError(
            "CUDA is already initialized without the requested cuBLAS workspace; start a fresh process"
        )
    if before != effective:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = effective
    return dict(
        device=device,
        before=before,
        effective=effective,
        requested=requested,
        action="set_before_cuda_initialization" if before != effective else "preserved",
        cuda_initialized_before=initialized,
    )


def _file(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def train_selector(manifest, output, *, resume=False, cublas_workspace=None):
    manifest = Path(manifest).resolve(strict=True)
    output = Path(output).resolve()
    definition = json.loads(manifest.read_text(encoding="utf-8"))
    device = definition.get("protocol", {}).get("device")
    source_root = Path(__file__).resolve().parents[1]
    launch = output.with_name(output.name + ".selector_launches") / (
        "%s_%s" % (time.time_ns(), uuid.uuid4().hex)
    )
    launch.mkdir(parents=True, exist_ok=False)  # Deliberately outside output.
    command = [
        sys.executable,
        "-m",
        "loris.discovery",
        "train-selector",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    ]
    if resume:
        command.append("--resume")
    if cublas_workspace is not None:
        command += ["--cublas-workspace", cublas_workspace]
    sources = [
        _file(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("__main__.py"),
            Path(__file__).with_name("selector_backend.py"),
            source_root / "_learning/selection.py",
        )
    ]
    started = dict(
        schema="loris_deterministic_selector_launch_v1",
        status="started",
        manifest=_file(manifest),
        output=str(output),
        resume=bool(resume),
        device=device,
        requested_workspace=cublas_workspace,
        workspace_before=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        command=command,
        sources=sources,
        python=sys.version,
        pid=os.getpid(),
        delegate="loris.discovery.selector_backend.run_selector_manifest",
        selector_algorithm_modified=False,
        selector_wrapper_extracted=True,
        resume_policy="same wrapper/source/input identity and original epoch checks; legacy outputs remain read-only",
    )
    _write(launch / "STARTED.json", started)
    begin = time.monotonic()
    workspace = None
    try:
        workspace = configure_workspace(device, cublas_workspace)
        from .selector_backend import run_selector_manifest

        result = run_selector_manifest(manifest, output, resume=resume)
        receipt = dict(
            status="complete",
            workspace=workspace,
            seconds=time.monotonic() - begin,
            complete=_file(output / "COMPLETE.json"),
            selected=_file(output / "SELECTED.json"),
            completed_optimizer_steps=result["completed_optimizer_steps"],
            selected_aliases_by_cluster=result["selected_aliases_by_cluster"],
        )
        _write(launch / "RESULT.json", receipt)
        return dict(receipt, launch_receipt=str(launch / "RESULT.json"))
    except BaseException as error:
        _write(
            launch / "RESULT.json",
            dict(
                status="failed",
                workspace=workspace,
                workspace_after=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                error_type=type(error).__name__,
                reason=str(error),
                seconds=time.monotonic() - begin,
                output_created=output.exists(),
                partial_outputs_preserved=True,
            ),
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="same selector wrapper, source, input and epoch identity required",
    )
    parser.add_argument(
        "--cublas-workspace",
        choices=WORKSPACES,
        help="CUDA: preserve valid environment or default to :4096:8; CPU unchanged",
    )
    args = parser.parse_args(argv)
    try:
        print(json.dumps(train_selector(**vars(args)), indent=2, allow_nan=False))
        return 0
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(dict(status="failed", error_type=type(error).__name__, reason=str(error))),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
