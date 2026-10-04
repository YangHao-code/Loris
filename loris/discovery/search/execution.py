"""Only compact-store import specialization; original search logic unchanged."""

import ast
import copy
import hashlib
import inspect
import textwrap
from pathlib import Path
from loris._engine import rlearner
from .sampler import source_record

STORAGE_MODULE = "loris.discovery.search.storage"


def _runner():
    source = textwrap.dedent(inspect.getsource(rlearner.run_cluster_search))
    tree = ast.parse(source)
    count = 0
    names = {
        "STORAGE_MODE",
        "FIXED_ATTR",
        "ParameterCodec",
        "CompactTrial",
        "SamplerCheckpointStore",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "loris._engine.compact_storage":
            if {n.name for n in node.names} != names:
                raise RuntimeError("upstream compact import differs")
            node.module = STORAGE_MODULE
            count += 1
    if count != 1:
        raise RuntimeError("expected one compact-store import")
    reverted = copy.deepcopy(tree)
    for node in ast.walk(reverted):
        if isinstance(node, ast.ImportFrom) and node.module == STORAGE_MODULE:
            node.module = "loris._engine.compact_storage"
    if ast.dump(reverted, include_attributes=False) != ast.dump(
        ast.parse(source), include_attributes=False
    ):
        raise RuntimeError("specialization altered search/scoring/transaction logic")
    namespace = dict(vars(rlearner))
    exec(
        compile(ast.fix_missing_locations(tree), "<rich_sampler_checkpoint_import_only>", "exec"),
        namespace,
    )
    binding = {
        "schema": "paper36_rich_sampler_import_specialization_v1",
        "original_function_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "original_module": source_record(rlearner.__file__),
        "adapter_source": source_record(Path(__file__).with_name("storage.py")),
        "specializer_source": source_record(__file__),
        "only_change": "one compact-store ImportFrom module; original globals not mutated",
    }
    return namespace["run_cluster_search"], binding


def execution_binding():
    return _runner()[1]


def run_rich_cluster_search(**kwargs):
    if kwargs.get("storage_mode", "compact_full_indicators_v1") != "compact_full_indicators_v1":
        raise ValueError("typed compact storage required")
    runner, binding = _runner()
    if kwargs.get("sampler_binding", {}).get("execution_specialization") != binding:
        raise ValueError("complete execution source binding required")
    return runner(**dict(kwargs, storage_mode="compact_full_indicators_v1"))
